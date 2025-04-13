import gradio as gr
from stainalyzer.trainer import Trainer

def run_stainalyzer(task, input_training_folder, training_severity, root_name):
    # Logic to route the task
    if task == "Staining":

        try:
            trainer = Trainer(
                training_image_path=input_training_folder,
                severity=training_severity,
                root_name=root_name
            )
            trainer.train(output_location=output_training_folder)
        except FileNotFoundError as fnf_error:
            print(f"Error: {fnf_error}")
        except ValueError as val_error:
            print(f"Error: {val_error}")
        except Exception as e:
            print(f"Unexpected error occurred: {e}")

    if task == "Segmentation":
        return "Hi"
    elif task == "Quantification":
        return "Hi"
    return "Task not implemented"

def launch_gui():
    interface = gr.Interface(
        fn=run_stainalyzer,
        inputs=[
            gr.File(file_types=[".tif", ".jpg", ".png", ".zip"], label="Upload images", file_count="multiple"),
            gr.Dropdown(["Staining", "Segmentation", "Quantification"], label="Select Task")
        ],
        outputs="text",
        title="Stainalyzer GUI",
        description="Lightweight batch analysis for smart biologists."
    )
    interface.launch()

