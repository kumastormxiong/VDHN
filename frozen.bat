d:
cd object_detection\
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ../training/ssd_mobilenet_v1_pets-vehicle.config --trained_checkpoint_prefix ../training/model.ckpt-524 --output_directory ../train_data