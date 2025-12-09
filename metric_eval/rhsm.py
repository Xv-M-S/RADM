import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 加载预训练的 VGG16 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
vgg16 = models.vgg16(pretrained=True).features.eval().to(device)

def preprocess_image(image_path):
    """
    预处理图像以适应 VGG16 输入要求
    
    :param image_path: str, 图像路径
    :return: tensor, 预处理后的图像
    """
    preprocess_vgg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    image_tensor = preprocess_vgg(image).unsqueeze(0).to(device)
    return image_tensor, width, height

def mask_image(image_tensor, layout_region, width, height):
    """
    在图像张量中应用掩码
    
    :param image_tensor: tensor, 图像张量
    :param layout_region: tuple, 布局区域 (x_min, y_min, x_max, y_max)
    :return: tensor, 掩码后的图像张量
    """
    masked_image = image_tensor.clone()
    h, w = masked_image.shape[2:]
    scale_x = w / width  # 假设原图宽度为 512
    scale_y = h / height  # 假设原图高度为 512
    scaled_layout_region = (
        int(layout_region[0] * scale_x),
        int(layout_region[1] * scale_y),
        int(layout_region[2] * scale_x),
        int(layout_region[3] * scale_y)
    )
    
    # 将布局区域外的部分设置为均值
    # mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    mean_tensor  = torch.full((1, 3, 1, 1), 0.5, device=device)
    masked_image[:, :, scaled_layout_region[1]:scaled_layout_region[3], scaled_layout_region[0]:scaled_layout_region[2]] = mean_tensor
    
    return masked_image

def calculate_feature_vector(image_tensor):
    """
    提取图像的特征向量
    
    :param image_tensor: tensor, 图像张量
    :return: tensor, 特征向量
    """
    with torch.no_grad():
        features = vgg16(image_tensor)
        features = features.view(features.size(0), -1)
    return features

def calculate_rshm(image_tensor, masked_image_tensor):
    """
    计算视觉平衡度 (Rshm)
    
    :param image_tensor: tensor, 原始图像张量
    :param masked_image_tensor: tensor, 掩码后的图像张量
    :return: float, Rshm值
    """
    feature_vector = calculate_feature_vector(image_tensor)
    masked_feature_vector = calculate_feature_vector(masked_image_tensor)
    
    # 计算 L2 距离
    l2_distance = torch.norm(feature_vector - masked_feature_vector, p=2).item()
    
    return l2_distance

if __name__ == "__main__":
    # 示例图像路径和布局区域
    image_path = "/home/sxm/flux-workspace/text-to-layout-zhuanlan/RADM/dataSets/RADM_dataset/images/train/O1CN01a0rOPI1aV1Bl3k4z0_!!3918463334-0-alimamazszw_mask002.jpg"
    layout_region = (100, 100, 400, 400)  # 示例布局区域 (x_min, y_min, x_max, y_max)
    # 示例使用
    image_tensor, width, height = preprocess_image(image_path)
    masked_image_tensor = mask_image(image_tensor, layout_region, width, height)

    rshm_value = calculate_rshm(image_tensor, masked_image_tensor)
    print(f'Rshm value: {rshm_value}')