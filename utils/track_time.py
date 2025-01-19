import time
import csv


class InferenceTimer:
    def __init__(self, model, dataset, csv_filename="inference_times.csv"):
        self.model = model
        self.dataset = dataset
        self.inference_times = []
        self.csv_filename = csv_filename
        self.sentences = []  # To store sentences in the order of inference

    def generate_motion_with_timing(self, sentence):
        start_time = time.time()
        # Replace 'generate' with the actual method name if different
        motion = self.model.generate(sentence)
        end_time = time.time()
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)
        self.sentences.append(sentence)
        return motion

    def calculate_average_time(self):
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    def save_to_csv(self):
        with open(self.csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Sentence", "Inference Time (s)"])
            for sentence, time in zip(self.sentences, self.inference_times):
                writer.writerow([sentence, time])
        print(f"Inference times saved to {self.csv_filename}")


# Example usage in generate.py or eval_humanml.py
if __name__ == "__main__":
    # Assume 'model' and 'dataset' are defined and loaded appropriately
    inference_timer = InferenceTimer(model, dataset)

    sentences = ["A person is walking.", "A person is running."]
    for sentence in sentences:
        motion = inference_timer.generate_motion_with_timing(sentence)
        # Further processing of 'motion' if needed

    # Calculate and print average inference time
    average_time = inference_timer.calculate_average_time()
    print(f"Average Inference Time per Sentence: {average_time:.4f} seconds")

    # Save inference times to CSV
    inference_timer.save_to_csv()
