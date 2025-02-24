import base64
from io import BytesIO
import torch
from torch import nn
from torchvision import transforms
import os

class StarterGenerator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64, num_classes=10):
        super(StarterGenerator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim + num_classes, hidden_dim * 8, kernel_size=4, stride=2),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2),
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True, stride=2),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise, labels):
        labels = nn.functional.one_hot(labels, self.num_classes).float().to(noise.device)
        combined_input = torch.cat((noise, labels), dim=1)
        x = combined_input.view(len(combined_input), -1, 1, 1)
        return self.gen(x)

class FollowerGenerator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(FollowerGenerator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2),
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, final_layer=True),
        )


    def make_gen_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

# Store loaded model so we don't need to reload it again
model_cache = {}

def load_model(name: str, z_dim:int=128, model_dir: str='./model'):
    # Return the stored model if exists
    if name in model_cache:
        return model_cache[name]
    
    if "starter" in name:
        model_path = os.path.join(model_dir, "starter", f'{name}.pt')
        generator = StarterGenerator(z_dim=z_dim)
    else:
        model_path = os.path.join(model_dir, "follower", f'{name}.pt')
        generator = FollowerGenerator(z_dim=z_dim)

    # For the sake of testing, We are only using starter models only.
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu')) # Load a model
        model_state_dict = checkpoint['generator_state_dict']
        generator.load_state_dict(model_state_dict) # Load the checkpoint
        generator.eval()  # Set to evaluation mode
        model_cache[name] = generator  # Cache the model
        print(f"Model for {name} loaded successfully.")
        return generator
    else:
        raise FileNotFoundError(f"Model for label '{name}' not found at {model_path}")

def generate(name: str, label: int=None):
    z_dim = 128
    z = torch.randn(1, z_dim)

    generator = load_model(name, z_dim)

    # Generate image
    with torch.no_grad():
        if label != None:
            generated_image = generator(z, label)
        else:
            generated_image = generator(z)

    # Rescale to [0, 1] range
    generated_image = (generated_image.squeeze() + 1) / 2

    # Convert to PIL Image and then to BytesIO
    pil_image = transforms.ToPILImage()(generated_image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")

    # Convert to base64
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    print("Finished printing an image with a label.")
    return base64_image


