import torch
import torch.nn.functional as F

class TransposeTensor(object):
    def __call__(self, tensor):
        # Transpose the tensor from (H, W, C) to (C, H, W)
        tensor = torch.transpose(tensor, 0, 2)
        tensor = torch.transpose(tensor, 1, 2)
        return tensor


class ResizeToCustomSize(object):
    def __init__(self, new_height, new_width):
        self.new_height = new_height
        self.new_width = new_width

    def __call__(self, tensor):
        # Compute the new height and width
        new_height = self.new_height
        new_width = self.new_width

        # Resize the tensor
        resized_tensor = F.interpolate(tensor.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)
        return resized_tensor


class GaussianDistributionTransform(object):
    def __init__(self, sigma, size):
        self.sigma = sigma
        self.size = size

    def __call__(self, xy):
        if xy == ('', ''):
            G = torch.zeros(self.size, device=torch.device('cuda'))
        else:
            xy = (int(xy[0]), int(xy[1]))
            xy = [item/2 for item in xy]
            xx, yy = torch.meshgrid(torch.arange(self.size[0], device=torch.device('cuda')), torch.arange(self.size[1], device=torch.device('cuda')))
            distances = ((xx - xy[0]) ** 2 + (yy - xy[1]) ** 2) / (2 * self.sigma ** 2)
            G = torch.exp(-distances) * 255
            G = torch.floor(G)
            #print(G.is_cuda)
            #del xx, yy, distances
        G = G.int()
        output = self.y_3d(G)
        #del G
        #torch.cuda.empty_cache()
        return output

    def y_3d(self, y):
        output_y = torch.zeros(256, 360, 640, device=torch.device('cuda'))
        index_tensor = torch.arange(256, device=torch.device('cuda')).view(-1, 1, 1)
        mask = index_tensor == y.unsqueeze(0)
        output_y[mask] = 1
        #del y, index_tensor, mask
        #torch.cuda.empty_cache()
        return output_y

class ToFloatTensor(object):
    def __call__(self, tensor):
        return tensor.float()

    [0, 0, 0, 0, 1, 0]
    4
