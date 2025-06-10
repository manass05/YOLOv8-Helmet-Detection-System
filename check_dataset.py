import os

# --- CONFIGURE THIS ---
# Path to the folder containing your 'train', 'valid', 'test' subfolders
dataset_path = 'Dataset/helmet detection.v1i.yolov8' # e.g., 'Helmet-Detection-1'
class_names = {0: 'With Helmet', 1: 'Without Helmet'} # IMPORTANT: Match your class numbers and names
# ----------------------

image_count = 0
instance_counts = {name: 0 for name in class_names.values()}
total_instances = 0

for split in ['train', 'valid', 'test']:
    image_folder = os.path.join(dataset_path, split, 'images')
    label_folder = os.path.join(dataset_path, split, 'labels')

    if os.path.exists(image_folder):
        image_count += len([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

    if os.path.exists(label_folder):
        for label_file in os.listdir(label_folder):
            if label_file.endswith('.txt'):
                with open(os.path.join(label_folder, label_file), 'r') as f:
                    for line in f:
                        total_instances += 1
                        class_id = int(line.split()[0])
                        class_name = class_names.get(class_id, 'unknown_class')
                        if class_name in instance_counts:
                            instance_counts[class_name] += 1

print(f"--- Dataset Stats ---")
print(f"Total Images: {image_count}")
print(f"Total Annotated Objects (Instances): {total_instances}")
print("\nInstances per Class:")
for name, count in instance_counts.items():
    print(f"  - {name}: {count}")
print("--------------------")