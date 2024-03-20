import torch


def jitter(image, jitter_rate=0.1):

    batch_size, num_channels, height, width = image.shape
    num_jitters = int(height * width * jitter_rate)

    jitter_indices = torch.randint(0, height * width, (batch_size, num_channels, num_jitters))
    jitter_magnitude = torch.randn((batch_size, num_channels, num_jitters))
    b, c = torch.meshgrid([torch.arange(0, batch_size), torch.arange(0, num_channels)])
    b = b.unsqueeze(-1).repeat(1, 1, num_jitters)
    c = c.unsqueeze(-1).repeat(1, 1, num_jitters)

    flattened_image = image.view(batch_size, num_channels, -1)
    flattened_image[b,c,jitter_indices] = jitter_magnitude
    jittered_image = flattened_image.view(batch_size, num_channels, height, width)

    return jittered_image


if __name__ == '__main__':

    batch_size = 3
    num_channels = 3
    height = 12
    width = 12

    image = torch.randn((batch_size, num_channels, height, width))
    jittered_image = jitter(image)
