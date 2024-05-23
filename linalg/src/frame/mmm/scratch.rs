use super::{BinOp, FusedKerSpec, FusedSpec, MatMatMulKer, OutputStoreKer};
use crate::LADatum;
use downcast_rs::{impl_downcast, Downcast};
use std::fmt::Debug;
use tract_data::internal::num_integer::Integer;
use tract_data::internal::*;

pub trait ScratchSpace: Downcast + Send {}
impl_downcast!(ScratchSpace);

#[derive(Debug, Default)]
pub struct ScratchSpaceImpl<TI: LADatum> {
    uspecs: Vec<FusedKerSpec<TI>>,
    blob: Blob,
    loc_dependant: TVec<LocDependant>,
    valid_down_tiles: usize,
    remnant_down: usize,
    valid_right_tiles: usize,
    remnant_right: usize,
}

#[derive(Debug, new)]
struct LocDependant {
    spec: usize,
    uspec: usize,
    // offset for the location dependant structure
    loc: usize,
    // offset of its associated dynamic-size buffers
    buffer_a: Option<usize>,
    buffer_b: Option<usize>,
}

impl<TI: LADatum> ScratchSpace for ScratchSpaceImpl<TI> {}
unsafe impl<TI: LADatum> Send for ScratchSpaceImpl<TI> {}

#[derive(Debug)]
struct AddMatMulTemp {
    ptr_a: *const u8,
    panel_a_id: usize,
    ptr_b: *const u8,
    panel_b_id: usize,
}

impl<TI: LADatum> ScratchSpaceImpl<TI> {
    pub unsafe fn prepare<K: MatMatMulKer<TI>>(
        &mut self,
        m: usize,
        n: usize,
        specs: &[FusedSpec],
    ) -> TractResult<()> {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        self.uspecs.clear();
        self.loc_dependant.clear();
        self.uspecs.reserve(specs.len() + 2);
        self.uspecs.push(FusedKerSpec::Clear);
        self.valid_down_tiles = m / K::mr();
        self.remnant_down = m % K::mr();
        self.valid_right_tiles = n / K::nr();
        self.remnant_right = n % K::nr();
        let mut offset = 0;
        let mut align = std::mem::size_of::<*const ()>();
        fn ld(spec: usize, uspec: usize, loc: usize) -> LocDependant {
            LocDependant { spec, uspec, loc, buffer_a: None, buffer_b: None }
        }
        for (ix, spec) in specs.iter().enumerate() {
            let uspec = match spec {
                FS::BinScalar(t, op) => match op {
                    BinOp::Min => FKS::ScalarMin(*t.to_scalar()?),
                    BinOp::Max => FKS::ScalarMax(*t.to_scalar()?),
                    BinOp::Mul => FKS::ScalarMul(*t.to_scalar()?),
                    BinOp::Add => FKS::ScalarAdd(*t.to_scalar()?),
                    BinOp::Sub => FKS::ScalarSub(*t.to_scalar()?),
                    BinOp::SubF => FKS::ScalarSubF(*t.to_scalar()?),
                },
                FS::ShiftLeft(s) => FKS::ShiftLeft(*s),
                FS::RoundingShiftRight(s, rp) => FKS::RoundingShiftRight(*s, *rp),
                FS::QScale(s, rp, m) => FKS::QScale(*s, *rp, *m),
                FS::BinPerRow(_, _) => {
                    self.loc_dependant.push(ld(ix, self.uspecs.len(), offset));
                    offset += TI::datum_type().size_of() * K::mr();
                    FusedKerSpec::Done
                }
                FS::BinPerCol(_, _) => {
                    self.loc_dependant.push(ld(ix, self.uspecs.len(), offset));
                    offset += TI::datum_type().size_of() * K::nr();
                    FusedKerSpec::Done
                }
                FS::AddRowColProducts(_, _) => {
                    self.loc_dependant.push(ld(ix, self.uspecs.len(), offset));
                    offset += TI::datum_type().size_of() * (K::mr() + K::nr());
                    FusedKerSpec::Done
                }
                FS::Store(_) | FS::AddUnicast(_) => {
                    self.loc_dependant.push(ld(ix, self.uspecs.len(), offset));
                    offset += TI::datum_type().size_of() * K::mr() * K::nr();
                    FusedKerSpec::Done
                }
                FS::LeakyRelu(t) => FKS::LeakyRelu(*t.to_scalar()?),
                FS::AddMatMul { a, b, .. } => {
                    let mut ld = ld(ix, self.uspecs.len(), offset);
                    offset += std::mem::size_of::<AddMatMulTemp>();
                    if let Some(tmp) = a.scratch_panel_buffer_layout() {
                        align = tmp.align().lcm(&align);
                        offset = Integer::next_multiple_of(&offset, &tmp.align());
                        ld.buffer_a = Some(offset);
                        offset += tmp.size();
                    }
                    if let Some(tmp) = b.scratch_panel_buffer_layout() {
                        align = tmp.align().lcm(&align);
                        offset = Integer::next_multiple_of(&offset, &tmp.align());
                        ld.buffer_b = Some(offset);
                        offset += tmp.size();
                    }
                    self.loc_dependant.push(ld);
                    FusedKerSpec::Done
                }
            };
            self.uspecs.push(uspec);
        }
        self.uspecs.push(FKS::Done);

        self.blob.ensure_size_and_align(offset, align);
        for LocDependant { loc, spec, .. } in &mut self.loc_dependant {
            let spec = specs.get_unchecked(*spec);
            #[allow(clippy::single_match)]
            match spec {
                FS::AddMatMul { .. } => {
                    let scratch = &mut *(self.blob.as_ptr().add(*loc) as *mut AddMatMulTemp);
                    scratch.panel_a_id = usize::MAX;
                    scratch.panel_b_id = usize::MAX;
                }
                _ => (),
            };
        }
        Ok(())
    }

    pub unsafe fn run<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
    ) {
        if down < self.valid_down_tiles && right < self.valid_right_tiles {
            self.for_valid_tile::<K>(specs, down, right);
            let err = K::kernel(self.uspecs());
            debug_assert_eq!(err, 0, "Kernel return error {err}");
        } else {
            let remnant_down =
                if down < self.valid_down_tiles { K::mr() } else { self.remnant_down };
            let remnant_right =
                if right < self.valid_right_tiles { K::nr() } else { self.remnant_right };
            self.for_border_tile::<K>(specs, down, right, remnant_down, remnant_right);
            let err = K::kernel(self.uspecs());
            debug_assert_eq!(err, 0, "Kernel return error {err}");
            self.postprocess_tile::<K>(specs, down, right, remnant_down, remnant_right);
        }
    }

    #[inline(always)]
    pub unsafe fn for_valid_tile<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
    ) {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        let ScratchSpaceImpl { uspecs, loc_dependant, .. } = self;
        debug_assert!(specs.len() + 2 == uspecs.len());
        for LocDependant { spec, uspec, loc, buffer_a, buffer_b } in loc_dependant {
            let spec = specs.get_unchecked(*spec);
            *uspecs.get_unchecked_mut(*uspec) = match spec {
                FS::BinPerRow(v, op) => {
                    let v = v.as_ptr_unchecked::<TI>().add(down * K::mr());
                    match op {
                        BinOp::Min => FKS::PerRowMin(v),
                        BinOp::Max => FKS::PerRowMax(v),
                        BinOp::Add => FKS::PerRowAdd(v),
                        BinOp::Mul => FKS::PerRowMul(v),
                        BinOp::Sub => FKS::PerRowSub(v),
                        BinOp::SubF => FKS::PerRowSubF(v),
                    }
                }
                FS::BinPerCol(v, op) => {
                    let v = v.as_ptr_unchecked::<TI>().add(right * K::nr());
                    match op {
                        BinOp::Min => FKS::PerColMin(v),
                        BinOp::Max => FKS::PerColMax(v),
                        BinOp::Add => FKS::PerColAdd(v),
                        BinOp::Mul => FKS::PerColMul(v),
                        BinOp::Sub => FKS::PerColSub(v),
                        BinOp::SubF => FKS::PerColSubF(v),
                    }
                }
                FS::AddRowColProducts(rows, cols) => {
                    let row_ptr = rows.as_ptr_unchecked::<TI>().add(down * K::mr());
                    let col_ptr = cols.as_ptr_unchecked::<TI>().add(right * K::nr());
                    FKS::AddRowColProducts(row_ptr, col_ptr)
                }
                FS::AddUnicast(store) => FKS::AddUnicast(store.tile_c(down, right)),
                FS::Store(c_store) => FKS::Store(c_store.tile_c(down, right)),
                FS::AddMatMul { a, b } => {
                    let scratch =
                        (self.blob.as_mut_ptr().add(*loc) as *mut AddMatMulTemp).as_mut().unwrap();
                    if scratch.panel_a_id != down {
                        scratch.ptr_a =
                            a.panel_bytes(down, buffer_a.map(|o| self.blob.as_mut_ptr().add(o)));
                        scratch.panel_a_id = down;
                    }
                    if scratch.panel_b_id != right {
                        scratch.ptr_b =
                            b.panel_bytes(right, buffer_b.map(|o| self.blob.as_mut_ptr().add(o)));
                        scratch.panel_b_id = right;
                    }
                    FKS::AddMatMul {
                        k: b.k(),
                        pa: scratch.ptr_a,
                        pb: scratch.ptr_b,
                        cpu_variant: 0,
                    }
                }
                _ => std::hint::unreachable_unchecked(),
            };
        }
    }

    #[inline(never)]
    pub unsafe fn for_border_tile<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
        m_remnant: usize,
        n_remnant: usize,
    ) {
        use FusedKerSpec as FKS;
        use FusedSpec as FS;
        let ScratchSpaceImpl { uspecs, loc_dependant, .. } = self;
        debug_assert!(specs.len() + 2 == uspecs.len());
        for LocDependant { spec, uspec, loc, buffer_a, buffer_b } in loc_dependant {
            let loc = self.blob.as_mut_ptr().add(*loc);
            let spec = specs.get_unchecked(*spec);
            *uspecs.get_unchecked_mut(*uspec) = match spec {
                FS::BinPerRow(v, op) => {
                    let buf = std::slice::from_raw_parts_mut(loc as *mut TI, K::mr());
                    let ptr = if m_remnant < K::mr() {
                        if m_remnant > 0 {
                            buf.get_unchecked_mut(..m_remnant).copy_from_slice(
                                v.as_slice_unchecked()
                                    .get_unchecked(down * K::mr()..)
                                    .get_unchecked(..m_remnant),
                            );
                        }
                        if cfg!(debug_assertions) {
                            buf.get_unchecked_mut(m_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
                        }
                        buf.as_ptr()
                    } else {
                        v.as_ptr_unchecked::<TI>().add(down * K::mr())
                    };
                    match op {
                        BinOp::Min => FKS::PerRowMin(ptr),
                        BinOp::Max => FKS::PerRowMax(ptr),
                        BinOp::Add => FKS::PerRowAdd(ptr),
                        BinOp::Mul => FKS::PerRowMul(ptr),
                        BinOp::Sub => FKS::PerRowSub(ptr),
                        BinOp::SubF => FKS::PerRowSubF(ptr),
                    }
                }
                FS::BinPerCol(v, op) => {
                    let buf = std::slice::from_raw_parts_mut(loc as *mut TI, K::nr());
                    let ptr = if n_remnant < K::nr() {
                        if n_remnant > 0 {
                            buf.get_unchecked_mut(..n_remnant).copy_from_slice(
                                v.as_slice_unchecked()
                                    .get_unchecked(right * K::nr()..)
                                    .get_unchecked(..n_remnant),
                            );
                        }
                        if cfg!(debug_assertions) {
                            buf.get_unchecked_mut(n_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
                        }
                        buf.as_ptr()
                    } else {
                        v.as_ptr_unchecked::<TI>().add(right * K::nr())
                    };
                    match op {
                        BinOp::Min => FKS::PerColMin(ptr),
                        BinOp::Max => FKS::PerColMax(ptr),
                        BinOp::Add => FKS::PerColAdd(ptr),
                        BinOp::Mul => FKS::PerColMul(ptr),
                        BinOp::Sub => FKS::PerColSub(ptr),
                        BinOp::SubF => FKS::PerColSubF(ptr),
                    }
                }
                FS::AddRowColProducts(rows, cols) => {
                    let r = std::slice::from_raw_parts_mut(loc as *mut TI, K::mr());
                    let row_ptr = if m_remnant < K::mr() {
                        r.get_unchecked_mut(..m_remnant).copy_from_slice(
                            rows.as_slice_unchecked()
                                .get_unchecked(down * K::mr()..)
                                .get_unchecked(..m_remnant),
                        );
                        if cfg!(debug_assertions) {
                            r.get_unchecked_mut(m_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
                        }
                        r.as_ptr()
                    } else {
                        rows.as_ptr_unchecked::<TI>().add(down * K::mr())
                    };
                    let c = std::slice::from_raw_parts_mut((loc as *mut TI).add(K::mr()), K::nr());
                    let col_ptr = if n_remnant < K::nr() {
                        c.get_unchecked_mut(..n_remnant).copy_from_slice(
                            cols.as_slice_unchecked()
                                .get_unchecked(right * K::nr()..)
                                .get_unchecked(..n_remnant),
                        );
                        if cfg!(debug_assertions) {
                            r.get_unchecked_mut(n_remnant..)
                                .iter_mut()
                                .for_each(|x| *x = TI::zero());
                        }
                        c.as_ptr()
                    } else {
                        cols.as_ptr_unchecked::<TI>().add(right * K::nr())
                    };
                    FKS::AddRowColProducts(row_ptr, col_ptr)
                }
                FS::AddUnicast(store) => {
                    let row_byte_stride = store.row_byte_stride;
                    let col_byte_stride = store.col_byte_stride;
                    let tile_offset = row_byte_stride * down as isize * K::mr() as isize
                        + col_byte_stride * right as isize * K::nr() as isize;
                    let tile_ptr = store.ptr.offset(tile_offset);
                    let tmp_d_tile =
                        std::slice::from_raw_parts_mut(loc as *mut TI, K::mr() * K::nr());
                    if cfg!(debug_assertions) {
                        tmp_d_tile.iter_mut().for_each(|t| *t = TI::zero());
                    }
                    for r in 0..m_remnant as isize {
                        for c in 0..n_remnant as isize {
                            let inner_offset = c * col_byte_stride + r * row_byte_stride;
                            if inner_offset + tile_offset
                                < (store.item_size * store.item_count) as isize
                            {
                                *tmp_d_tile.get_unchecked_mut(r as usize + c as usize * K::mr()) =
                                    *(tile_ptr.offset(inner_offset) as *const TI);
                            }
                        }
                    }
                    FKS::AddUnicast(OutputStoreKer {
                        ptr: tmp_d_tile.as_ptr() as _,
                        row_byte_stride: std::mem::size_of::<TI>() as isize,
                        col_byte_stride: (std::mem::size_of::<TI>() * K::mr()) as isize,
                        item_size: std::mem::size_of::<TI>(),
                    })
                }
                FS::Store(c_store) => {
                    let tmpc = OutputStoreKer {
                        ptr: loc as _,
                        item_size: c_store.item_size,
                        row_byte_stride: c_store.item_size as isize,
                        col_byte_stride: (c_store.item_size * K::mr()) as isize,
                    };
                    FKS::Store(tmpc)
                }
                FS::AddMatMul { a, b } => {
                    let scratch = (loc as *mut AddMatMulTemp).as_mut().unwrap();
                    if scratch.panel_a_id != down {
                        scratch.ptr_a =
                            a.panel_bytes(down, buffer_a.map(|o| self.blob.as_mut_ptr().add(o)));
                        scratch.panel_a_id = down;
                    }
                    if scratch.panel_b_id != right {
                        scratch.ptr_b =
                            b.panel_bytes(right, buffer_b.map(|o| self.blob.as_mut_ptr().add(o)));
                        scratch.panel_b_id = right;
                    }
                    FKS::AddMatMul {
                        k: b.k(),
                        pa: scratch.ptr_a,
                        pb: scratch.ptr_b,
                        cpu_variant: 0,
                    }
                }
                _ => std::hint::unreachable_unchecked(),
            };
        }
    }

    #[inline]
    pub fn uspecs(&self) -> &[FusedKerSpec<TI>] {
        &self.uspecs
    }

    pub unsafe fn postprocess_tile<K: MatMatMulKer<TI>>(
        &mut self,
        specs: &[FusedSpec],
        down: usize,
        right: usize,
        m_remnant: usize,
        n_remnant: usize,
    ) where
        TI: LADatum,
    {
        for LocDependant { spec, uspec, .. } in self.loc_dependant.iter() {
            let spec = specs.get_unchecked(*spec);
            let ker_spec = self.uspecs.get_unchecked(*uspec);
            if let (FusedSpec::Store(c_store), FusedKerSpec::Store(tmp)) = (spec, ker_spec) {
                c_store.set_from_tile(down, right, m_remnant, n_remnant, tmp)
            }
        }
    }
}
