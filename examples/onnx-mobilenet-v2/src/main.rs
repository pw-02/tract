use std::fs::File;
use std::io::{BufWriter, Write};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};

use tract_onnx::{prelude::*, tract_core};
// use onnx::ModelProto;
use tract_onnx::tract_core::ops::submodel::InnerModel;
use tract_core::ops::konst::Const;

#[derive(Serialize, Deserialize)]
struct TensorData {
    // name: String,
    data: Vec<f32>,
    shape: Vec<usize>,
}


fn main() -> TractResult<()> {
    
    let mut model = tract_onnx::onnx().model_for_path("examples\\onnx-mobilenet-v2\\simple_cnn.onnx")?;
    let mut typed_model = model
        .into_typed()?
        // .concretize_dims(&symbol_values)?
        .into_decluttered()?;
    save_debug_info("examples\\onnx-mobilenet-v2\\typed_model_debug.txt", &typed_model)?;

    // let tensors = model.nodes().to_vec();
    // // Serialize tensors to JSON
    // let json_data = serde_json::to_string(&tensors)?;

    //  // Save JSON to file
    //  let mut file = File::create("tensors.json")?;
    //  file.write_all(json_data.as_bytes())?;
     //println!("{:?}", tensors);

    //let parse_Result =  tract_onnx::onnx().parse(model).unwrap();

    // // Define the path where you want to save the optimized ONNX model
    // let optimized_model_path = "examples\\onnx-mobilenet-v2\\optimized_mobilenetv2.onnx";
    // // Save the optimized model to a file
    
    // // Write the model data to a file
    // let mut writer = BufWriter::new(File::create(optimized_model_path)?);
    // writer.write_all(&model_data)?;

    Ok(())
}

fn save_debug_info<P: AsRef<Path>>(file_path: P, typed_model: &TypedModel) -> std::io::Result<()> {
    let debug_info = format!("{:?}", typed_model);
    let mut file = File::create(file_path)?;
    file.write_all(debug_info.as_bytes())?;
    Ok(())
}