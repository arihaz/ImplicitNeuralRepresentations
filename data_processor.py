import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


def decode_output_tensor(out_tensor, sidelen):
  return ((out_tensor.cpu().view(sidelen,sidelen,3).detach().numpy()*0.5)+0.5)*255

def rescale_image_array(img):
  return ((img / 255 ) - 0.5) / 0.5

def get_input_coords(sidelen, dim=2, multiple_images=True):
    '''Generates coordinates in range -1 to 1.
    sidelen: int
    dim: int
    multiple_images: bool - True to add an index in {-1,1} if there are 2 images
    '''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    if multiple_images:
      img_index = torch.ones(mgrid.shape[0])
      img_index = torch.unsqueeze(img_index, dim=1)

      mgrid = torch.cat([torch.cat([mgrid, img_index], dim=1), 
                        torch.cat([mgrid, -1*img_index], dim=1)], dim=0)
    return mgrid

def get_images_as_tensor(sidelength, img1, img2):
  #reshaping and normalizing in range [-1,1]
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
    ])
    img1= transform(img1)
    img2 = transform(img2)
    return img1, img2

class ImagesINR(Dataset):
    def __init__(self, sidelength, img1, img2):
        super().__init__()
        img1, img2 = get_images_as_tensor(sidelength, img1, img2)
        self.pixels = torch.cat([img1.reshape(3, -1).transpose(0, 1), 
                                 img2.reshape(3, -1).transpose(0, 1)], dim=0)
        self.coords = get_input_coords(sidelength, 2)

    def __len__(self):
        # all the coords are processed during one iteration
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels