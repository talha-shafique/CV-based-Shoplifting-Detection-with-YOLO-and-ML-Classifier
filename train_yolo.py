from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model on our custom dataset
if __name__ == '__main__':
    print("Starting YOLO model fine-tuning...")
    # Note: This process will take time and requires a capable machine.
    # It will automatically download dependencies and start training.
    results = model.train(
        data=r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\data.yaml',
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image size
        batch=8,    # Batch size
        name='yolo_shoplifting_custom' # Name for the training run
    )
    print("\nTraining complete.")
    print("The best model is saved in the 'runs/detect/yolo_shoplifting_custom/weights' directory.")
