import os

def invert_yolo_labels(label_folder):
    if not os.path.isdir(label_folder):
        print(f"Directory not found: {label_folder}")
        return
    print("Directory contents:", os.listdir(label_folder))


    files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    print(files)
    print(f"Processing {len(files)} label files in: {label_folder}")

    for file in files:
        file_path = os.path.join(label_folder, file)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = parts[0]

            if class_id == '0':
                parts[0] = '1'
            elif class_id == '1':
                parts[0] = '0'

            new_lines.append(' '.join(parts) + '\n')

        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    print("Label inversion complete.")



if __name__ == "__main__":
    label_dir_train = r"C:\Users\aswan\College Project\Le2i_augmented_dataset\labels\train"
    invert_yolo_labels(label_dir_train)
