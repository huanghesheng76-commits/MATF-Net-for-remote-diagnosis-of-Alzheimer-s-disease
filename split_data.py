import os
import random
import shutil
import time


root_dir = 'C:\\Alzheimer\\ADNI1_1'
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')
test_dir = os.path.join(root_dir, 'test')
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2


def create_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)


def get_image_paths(root_dir):
    perspectives = ['end-AP', 'end-CP', 'end-SP']
    classes = ['P-NC', 'P-MCI', 'P-AD']
    image_paths = {
        perspective: {class_name: [] for class_name in classes}
        for perspective in perspectives
    }

    for perspective in perspectives:
        perspective_dir = os.path.join(root_dir, perspective)
        for class_name in classes:
            class_dir = os.path.join(perspective_dir, class_name)
            image_paths[perspective][class_name] = sorted(
                [
                    os.path.join(class_dir, img)
                    for img in os.listdir(class_dir)
                    if img.endswith('.png')
                ]
            )

    return image_paths


image_paths = get_image_paths(root_dir)


random.seed(1)
random_sequences_cp = {
    class_name: random.sample(
        range(len(image_paths['end-CP'][class_name])),
        len(image_paths['end-CP'][class_name])
    )
    for class_name in ['P-NC', 'P-MCI', 'P-AD']
}
random_sequences_ap = random_sequences_cp
random_sequences_sp = random_sequences_cp

random_sequences = {
    'end-SP': random_sequences_sp,
    'end-CP': random_sequences_cp,
    'end-AP': random_sequences_ap
}

for perspective, classes in image_paths.items():
    for class_name, images in classes.items():
        image_paths[perspective][class_name] = [
            images[idx] for idx in random_sequences[perspective][class_name]
        ]


num_train = {
    perspective: {
        class_name: int(len(images) * train_ratio)
        for class_name, images in classes.items()
    }
    for perspective, classes in image_paths.items()
}
num_val = {
    perspective: {
        class_name: int(len(images) * val_ratio)
        for class_name, images in classes.items()
    }
    for perspective, classes in image_paths.items()
}
num_test = {
    perspective: {
        class_name: len(images)
        - num_train[perspective][class_name]
        - num_val[perspective][class_name]
        for class_name, images in classes.items()
    }
    for perspective, classes in image_paths.items()
}


print("Number of images for training, validation, and test sets:")
print(num_train)
print(num_val)
print(num_test)


end_test_indices = {
    perspective: {
        class_name: len(images)
        for class_name, images in classes.items()
    }
    for perspective, classes in image_paths.items()
}

end_val_indices = {
    perspective: {
        class_name: num_train[perspective][class_name]
        + num_val[perspective][class_name]
        for class_name in classes
    }
    for perspective, classes in image_paths.items()
}


def copy_images(image_paths, dest_dir, start_indices, end_indices):
    for perspective, classes in image_paths.items():
        for class_name, images in classes.items():
            start_index = start_indices[perspective][class_name]
            end_index = end_indices[perspective][class_name]
            for img in images[start_index:end_index]:
                dest_subdir = os.path.join(dest_dir, perspective, class_name)
                os.makedirs(dest_subdir, exist_ok=True)
                shutil.copy(
                    img,
                    os.path.join(dest_subdir, os.path.basename(img))
                )


copy_images(
    image_paths,
    train_dir,
    {
        perspective: {
            class_name: 0 for class_name in classes
        }
        for perspective, classes in image_paths.items()
    },
    num_train
)

copy_images(
    image_paths,
    val_dir,
    {
        perspective: {
            class_name: num_train[perspective][class_name]
            for class_name in classes
        }
        for perspective, classes in image_paths.items()
    },
    end_val_indices
)

copy_images(
    image_paths,
    test_dir,
    {
        perspective: {
            class_name: num_train[perspective][class_name]
            + num_val[perspective][class_name]
            for class_name in classes
        }
        for perspective, classes in image_paths.items()
    },
    end_test_indices
)