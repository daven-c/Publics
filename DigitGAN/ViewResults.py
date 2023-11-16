from TorchModels import Generator
from matplotlib import pyplot as plt
import torch


def hot_encode_num(num: int):
    vals = {
        0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    return vals[num]


def multi_one_hot_label(labels):
    classes = 10
    one_hot = torch.zeros(labels.size(0), classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)

    return one_hot


def load_configs(path: str = "settings.cfg"):
    with open(path, "r") as f:
        lines = f.readlines()
        settings = {}
        for line in lines:
            data = line.rstrip().split("=")
            if len(data) == 2:
                if data[1].count(".") > 0:  # Floats
                    data[1] = float(data[1])
                elif data[1].isdigit():  # Ints
                    data[1] = int(data[1])
                elif data[1] == "True":  # Bool
                    data[1] = True
                elif data[1] == "False":
                    data[1] = False
                settings.update({data[0]: data[1]})
        return settings


def plot_one_image(img):
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


def draw(model, num, noise_amount):
    if not 0 <= num < 10:
        return
    noise = torch.rand(size=(1, noise_amount))
    label = torch.tensor(hot_encode_num(num)).view(1, 10)
    generated_image = model.predict(noise, label)
    plot_one_image(generated_image)


def preview_all(model, noise_amount, return_only: bool = False):
    noise = torch.rand(size=(10, noise_amount))
    labels = torch.tensor([hot_encode_num(i) for i in range(10)])
    generated_images = model.predict(
        noise, labels).detach().numpy()

    plot = plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.title(i)
        plt.imshow(generated_images[i],
                   interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()

    if not return_only:
        plt.show()

    return plot


if __name__ == "__main__":
    filename = "GAN_11-15-23_19.56"

    configs = load_configs(f"./models/{filename}/settings.cfg")
    num_classes = 10
    noise_amount = configs["noise_amount"]
    epochs = configs["epochs"]
    save_interval = configs["save_interval"]
    batch_size = configs["batch_size"]
    learning_rate = configs["learning_rate"]
    beta1 = configs["beta_1"]
    skip_preview = configs["skip_preview"]

    generator = Generator(input_size=noise_amount +
                          num_classes, hidden_dim=256, output_size=28 * 28, lr=learning_rate, beta1=beta1)
    generator.load_state_dict(torch.load(f"./models/{filename}/GenModel.pth"))
    generator.eval()
    # generator.summary()

    preview_all(generator, configs["noise_amount"])
    inp = ""
    while inp != "exit":
        inp = input("number: ")
        if not inp.isdigit():
            continue
        inp = int(inp)
        draw(generator, inp, configs["noise_amount"])
