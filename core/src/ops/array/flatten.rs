use crate::internal::*;

// TODO: canonicalize as Reshape

#[derive(Debug, Clone, new, Default)]
pub struct Flatten {
    axis: usize,
}

impl Flatten {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(
        &self,
        input: Arc<Tensor>,
        shape: (usize, usize),
    ) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec![input.into_tensor().into_array::<T>()?.into_shape(shape)?.into_arc_tensor()])
    }

    pub fn compute_shape<D: DimLike>(&self, shape: &[D]) -> TractResult<[D; 2]> {
        if shape.iter().filter(|d| d.to_integer().is_err()).count() > 1 {
            bail!("Can not compute a shape with square of symbols")
        }
        let shape_left: usize =
            shape[..self.axis].iter().fold(1, |acc, v| acc * v.to_integer().unwrap_or(1)) as usize;
        let shape_right: usize =
            shape[self.axis..].iter().fold(1, |acc, v| acc * v.to_integer().unwrap_or(1)) as usize;
        let stream = shape.iter().position(|d| d.to_integer().is_err());
        Ok(match stream {
            None => [D::from(shape_left), D::from(shape_right)],
            Some(ix) if ix < self.axis => [shape[ix].clone() * shape_left, shape_right.into()],
            Some(ix) => [shape_left.into(), shape[ix].clone() * shape_right],
        })
    }
}

impl Op for Flatten {
    fn name(&self) -> Cow<str> {
        "Flatten".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Flatten {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let [shape_0, shape_1] = self.compute_shape(input.shape())?;
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input, (shape_0, shape_1)))
    }
}

impl TypedOp for Flatten {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            self.compute_shape(&*inputs[0].shape.to_tvec())?.as_ref(),
        )?))
    }
}
