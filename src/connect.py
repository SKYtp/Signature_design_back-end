import base64
from io import BytesIO
import cv2
import torch
from torch import nn
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from src.image_to_value import*
from src.get_contour import*
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_to_base64(image):
    # แปลงรูปเป็น buffer (JPEG format)
    _, buffer = cv2.imencode('.jpg', image)

    # แปลง buffer เป็น Base64
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return base64_image

def is_broken_line(image,c):
    edges = cv2.Canny(image, 50, 150)

    # หาคอนทัวร์
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ถ้ามีหลายเส้นขอบ แปลว่ามีเส้นขาด
    if len(contours) > 1:
        if c=='ษ' and len(contours) == 2:
            return False
        print('เส้นขาด')
        return True  # มีเส้นขาด
    else:
        return False  # เส้นไม่ขาด
    
def get_text_height(image):
    # ใช้ Threshold เพื่อแปลงภาพเป็นไบนารี่
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # ค้นหา Contours (เส้นขอบของตัวอักษร)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # คำนวณ Bounding Box
        
    return h

def rotate_image(image, angle):
    
    # ได้ขนาดรูปภาพ
    (h, w) = image.shape[:2]

    # คำนวณจุดศูนย์กลางของรูปภาพ
    center = (w // 2, h // 2)

    # คำนวณเมทริกซ์หมุน
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # หมุนภาพ
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))

    return rotated

def compare_text_height(image1, image2):
    height1 = get_text_height(image1)
    height2 = get_text_height(image2)

    x = height2/height1
    if x > 0.40 and x < 0.65:
        print(f'ความสูงผ่าน{x}')
        return False , x
    else:
        print(f'ความสูงไม่ได้{x}')
        return True , x

def add_black_dots(image, points, radius=3):
    # วาดจุดสีดำตามที่กำหนด
    for (x, y) in points:
        cv2.circle(image, (x, y), radius, (0, 0, 0), -1, cv2.LINE_AA)  # -1 หมายถึงเติมเต็มวงกลม

    return image

def draw_diagonal_line(image, start_point, length, thickness=2, color=(0, 0, 0)):
    # คำนวณจุดปลาย (เอียง 45 องศา)
    end_point = (start_point[0] + length, start_point[1] - length)

    # วาดเส้น
    cv2.line(image, start_point, end_point, color, thickness, cv2.LINE_AA)

    return image , end_point

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

        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

# Store loaded model so we don't need to reload it again
model_cache = {}


def load_model(position,name: str, model_dir: str='./model'):
    z_dim:int=128
    
    if position==0:
        model_path = os.path.join(model_dir, "starter", f'{name}_checkpoint_epoch_4999.pt')
        generator = StarterGenerator(z_dim=z_dim)
    elif position==1:
        model_path = os.path.join(model_dir, "follower", f'{name}_model.pt')
        generator = FollowerGenerator(z_dim=z_dim)
    else:
        model_path = os.path.join(model_dir, "end", f'{name}_back_model.pt')
        generator = FollowerGenerator(z_dim=z_dim)

    # For the sake of testing, We are only using starter models only.
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'),weights_only=True) # Load a model
        model_state_dict = checkpoint['generator_state_dict']
        generator.load_state_dict(model_state_dict) # Load the checkpoint
        generator.eval()  # Set to evaluation mode
        model_cache[name] = generator  # Cache the model
        print(f"Model for {name} loaded successfully.")
        return generator
    else:
        raise FileNotFoundError(f"Model for label '{name}' not found at {model_path}")

def generate(position, generator, label: int=None,):
    z_dim = 128
    z = torch.randn(1, z_dim, device=device)

    # Generate image
    with torch.no_grad():
        if position == 0:
            generated_image = generator(z, label)
        else:
            generated_image = generator(z)

    # Rescale to [0, 1] range
    generated_image = (generated_image.squeeze() + 1) / 2

    # Convert to PIL Image and then to BytesIO
    pil_image = transforms.ToPILImage()(generated_image)

    print("Finished printing an image with a label.")
    return pil_image

def rgb2gray(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
def genImage_for_one(name, style):
    model_path = r"./model"
    f_pos = ['ว','ส','ป']
    image=[]
    h_ratio = None
    omega = None
    loop = None
    consonant=[]
    
    if name in f_pos:
        consonant.append(name)

    if consonant[0] in f_pos:
        c = consonant[0]
        generator = load_model(0,c,model_path).to(device)
        pic = rgb2gray(generate(0,generator,torch.tensor([style])))
        while final_value_contour_head(pic)[0]:
            pic = rgb2gray(generate(0,generator,torch.tensor([style])))
            
        head = final_value_contour_head(pic)
        image.append(pic)
    return image , consonant , omega , loop , head ,  h_ratio


def genImage(name, style, nothing_omega_loop):
    model_path = r"./model"
    f_pos = ['ว','ส','ป']
    s_pos = ['ก','ข','ฃ','ค','ฅ','ฆ','ง','จ','ฉ','ช','ซ','ฌ','ญ','ฎ','ฏ','ฐ','ฑ','ฒ','ณ','ด','ต','ถ','ท','ธ','น','บ','ป','ผ','ฝ','พ','ฟ','ภ','ม','ย','ร','ล','ว','ศ','ษ','ส','ห','ฬ','อ','ฮ']
    s_pos_top = ['ข','ฃ','ฆ','ช','ซ','ฌ','ญ','ฐ','ฒ','ณ','ธ','น','บ','ป','ผ','ฝ','พ','ฟ','ม','ย','ร','ศ','ษ','ส','ฬ','ฮ']
    s_pos_bottom = ['ก','ค','ฅ','ฑ','ด','ต','ถ','ท','ภ','ห']
    l_pos = ['ก','ข','ฃ','ค','ฅ','ฆ','ง','จ','ฉ','ช','ซ','ฌ','ญ','ฎ','ฏ','ฐ','ฑ','ฒ','ณ','ด','ต','ถ','ท','ธ','น','บ','ป','ผ','ฝ','พ','ฟ','ภ','ม','ย','ร','ล','ว','ศ','ษ','ส','ห','ฬ','อ','ฮ']
    l_pos_startTop = ['ข','ฃ','ฆ','ช','ซ','ฑ','ท','ธ','ฐ','น','บ','ป','ผ','ฝ','พ','ฟ','ม','ย','ษ','ห','ฬ','อ','ฮ']    
    image=[]
    h_ratio = None
    omega = None
    loop = None
    consonant=[]
    for i in name:
        if i in s_pos:
            consonant.append(i)
            
    if consonant[0] in f_pos:
        c = consonant[0]
        generator = load_model(0,c,model_path).to(device)
        pic = rgb2gray(generate(0,generator,torch.tensor([style])))
        while final_value_contour_head(pic)[0]:
            pic = rgb2gray(generate(0,generator,torch.tensor([style])))
            
        head = final_value_contour_head(pic)
        image.append(pic)
        if consonant[1] in s_pos:
            c = consonant[1]
            generator = load_model(1,c,model_path).to(device)
            pic = rgb2gray(generate(1,generator))
            while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                pic = rgb2gray(generate(1,generator))
            
            h_ratio = compare_text_height(image[0],pic)[1]
            image.append(pic)
            if nothing_omega_loop == 0:
                for i in range(len(consonant)-2):
                    c = consonant[i+2]
                    generator = load_model(1,c,model_path).to(device)
                    pic = rgb2gray(generate(1,generator))
                    while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                        pic = rgb2gray(generate(1,generator))
                        
                    image.append(pic)

                return image , consonant , omega , loop , head ,  h_ratio
            else:
                for i in range(len(consonant)-3):    
                    if nothing_omega_loop == 1:
                        if consonant[1] in s_pos_top:
                            c = 'โอเมก้าหงาย'
                            generator = load_model(3,c,model_path).to(device)
                            pic = rgb2gray(generate(3,generator))
                            while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                                pic = rgb2gray(generate(3,generator))
                            omega = 0
                            image.append(pic)
    
                        elif consonant[1] in s_pos_bottom:
                            c = 'โอเมก้าคว่ำ'
                            generator = load_model(3,c,model_path).to(device)
                            pic = rgb2gray(generate(3,generator))
                            while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                                pic = rgb2gray(generate(3,generator))
                            omega = 1
                            image.append(pic)
                        elif consonant[-1] in l_pos_startTop:
                            c = 'โอเมก้าหงาย'
                            generator = load_model(3,c,model_path).to(device)
                            pic = rgb2gray(generate(3,generator))
                            while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                                pic = rgb2gray(generate(3,generator))
                            omega = 0
                            image.append(pic)
                        else:
                            c = 'โอเมก้าคว่ำ'
                            generator = load_model(3,c,model_path).to(device)
                            pic = rgb2gray(generate(3,generator))
                            while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                                pic = rgb2gray(generate(3,generator))
                            omega = 1
                            image.append(pic)
                    else:
                        if consonant[1] in s_pos_top:
                            c = 'บ่วงหงาย'
                            generator = load_model(3,c,model_path).to(device)
                            pic = rgb2gray(generate(3,generator))
                            while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                                pic = rgb2gray(generate(3,generator))
                            loop = 0
                            image.append(pic)
    
                        elif consonant[1] in s_pos_bottom:
                            c = 'บ่วงคว่ำ'
                            generator = load_model(3,c,model_path).to(device)
                            pic = rgb2gray(generate(3,generator))
                            while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                                pic = rgb2gray(generate(3,generator))
                            loop = 1
                            image.append(pic)
                        elif consonant[-1] in l_pos_startTop:
                            c = 'บ่วงหงาย'
                            generator = load_model(3,c,model_path).to(device)
                            pic = rgb2gray(generate(3,generator))
                            while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                                pic = rgb2gray(generate(3,generator))
                            loop = 0
                            image.append(pic)
                        else:
                            c = 'บ่วงคว่ำ'
                            generator = load_model(3,c,model_path).to(device)
                            pic = rgb2gray(generate(3,generator))
                            while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                                pic = rgb2gray(generate(3,generator))
                            loop = 1
                            image.append(pic)
                if consonant[-1] in l_pos:
                    c = consonant[-1]
                    generator = load_model(2,c,model_path).to(device)
                    pic = rgb2gray(generate(2,generator))
                    while is_broken_line(pic,c) or compare_text_height(image[0],pic)[0]:
                        pic = rgb2gray(generate(2,generator))
                        
                    image.append(pic)
                    return image , consonant , omega , loop , head ,h_ratio 
                else:
                    print("last position incorrect")
                    return None
        else:
            print("second position incorrect")
            return None
    else:
        print("first position incorrect")
        return None

def quadratic_bezier(t, p0, p1, p2):
    return (1-t)**2 * p0 + 2*(1-t)*t*p1 + t**2 * p2

def front_find_right_top(image):
    black_pixel_positions = np.column_stack(np.where(image <= 10))
    #หาบนขวาเริ่มขวา
    #หาจุดขวาสุด
    index_right=0
    for i, pos in enumerate(black_pixel_positions):
        if pos[1]>=index_right:
            index_right=pos[1]
            point_right=black_pixel_positions[i]
                            
    #หาจุดสูงสุดทางขวา
    index_top=125
    for i, pos in enumerate(black_pixel_positions):
        if pos[0]<=index_top and pos[1]>=point_right[1]-6:
            index_top=pos[0]
            point_top_right=black_pixel_positions[i]
    p0=np.array([point_top_right[1] ,point_top_right[0]])
    front_letter=0
    return p0 , front_letter

def front_find_right_bottom(image):
    black_pixel_positions = np.column_stack(np.where(image <= 10))
    #หาจุดขวาสุด
    index_right=0
    for i, pos in enumerate(black_pixel_positions):
        if pos[1]>=index_right:
            index_right=pos[1]
            point_right=black_pixel_positions[i]
                            
    #หาจุดต่ำสุด
    index_bottom=0
    for i, pos in enumerate(black_pixel_positions):
        if pos[0]>=index_bottom and pos[1]>=point_right[1]-3:
            index_bottom=pos[0]
            point_bottom_right=black_pixel_positions[i]
    p0=np.array([point_bottom_right[1],point_bottom_right[0]])
    front_letter=1
    return p0 , front_letter

def back_find_left_top(image, width):
    black_pixel_positions = np.column_stack(np.where(image <= 10))
    #หาจุดซ้ายสุด
    index_left=10000
    for i, pos in enumerate(black_pixel_positions):
        if pos[1]<=index_left:
            index_left=pos[1]
            point_left=black_pixel_positions[i]
    #หาจุดสูงสุด
    index_top=125
    for i, pos in enumerate(black_pixel_positions):
        if pos[0]<=index_top and pos[1]<=point_left[1]+2:
            index_top=pos[0]
            point_topleft=black_pixel_positions[i]
    p2=np.array([point_topleft[1]+width ,point_topleft[0]])
    back_letter=0
    return p2 , back_letter

def back_find_left_bottom(image, width):
    black_pixel_positions = np.column_stack(np.where(image <= 10))
    #หาจุดซ้ายสุด
    index_left=10000
    for i, pos in enumerate(black_pixel_positions):
        if pos[1]<=index_left:
            index_left=pos[1]
            point_left=black_pixel_positions[i]
    #หาจุดต่ำสุด
    index_bottom=0
    for i, pos in enumerate(black_pixel_positions):
        if pos[0]>=index_bottom and pos[1]<=point_left[1]+3:
            index_bottom=pos[0]
            point_bottomleft=black_pixel_positions[i]
    p2=np.array([point_bottomleft[1]+width ,point_bottomleft[0]])
    back_letter=1
    return p2 , back_letter

def back_find_top_left(image, width):
    black_pixel_positions = np.column_stack(np.where(image <= 10))
    #หาจุดสูงสุด
    index_top=125
    for i, pos in enumerate(black_pixel_positions):
        if pos[0]<=index_top:
            index_top=pos[0]
            point_top=black_pixel_positions[i]
    #หาจุดซ้ายสุด
    index_left=10000
    for i, pos in enumerate(black_pixel_positions):
        if pos[1]<=index_left and pos[0]<= point_top[0]+3:
            index_left=pos[1]
            point_topleft=black_pixel_positions[i]
    p2=np.array([point_topleft[1]+width ,point_topleft[0]])
    back_letter=0
    return p2 , back_letter

def back_find_bottom_left(image, width):
    black_pixel_positions = np.column_stack(np.where(image <= 10))
    #หาจุดต่ำสุด
    index_bottom = 0
    for i, pos in enumerate(black_pixel_positions):
        if pos[0]>=index_bottom:
            index_bottom=pos[0]
            point_bottom=black_pixel_positions[i]
    #หาจุดซ้ายสุด
    index_left=10000
    for i, pos in enumerate(black_pixel_positions):
        if pos[1]<=index_left and pos[0] >= point_bottom[0]-3:
            index_left=pos[1]
            point_bottomleft=black_pixel_positions[i]
    p2=np.array([point_bottomleft[1]+width ,point_bottomleft[0]])
    back_letter=1
    return p2 , back_letter

def back_find_left(image, width):
    black_pixel_positions = np.column_stack(np.where(image <= 10))
    #หาจุดซ้ายสุด
    index_left=10000
    for i, pos in enumerate(black_pixel_positions):
        if pos[1]<=index_left:
            index_left=pos[1]
            point_left=black_pixel_positions[i]
    p2=np.array([point_left[1]+width ,point_left[0]])
    back_letter = 0
    return p2 , back_letter

def back_find_left_mid_from_bottom(image, width):
    black_pixel_positions = np.column_stack(np.where(image <= 10))
    #หาจุดต่ำสุด
    index_bottom = 0
    for i, pos in enumerate(black_pixel_positions):
        if pos[0]>=index_bottom:
            index_bottom=pos[0]
            point_bottom=black_pixel_positions[i]
    #หาจุดซ้ายสุด
    index_left=10000
    for i, pos in enumerate(black_pixel_positions):
        if pos[1]<=index_left and pos[0] >= point_bottom[0]-6:
            index_left=pos[1]
            point_bottomleft=black_pixel_positions[i]
    #หาจุดสูงสุด
    index_top = point_bottomleft[0]
    index_left = point_bottomleft[1]
    print(index_top,index_left)
    for i, pos in enumerate(black_pixel_positions):
        if pos[0]<=index_top and pos[0]>=index_top-25 and pos[1]<=index_left+3:
            index_top=pos[0]
            point_top=black_pixel_positions[i]
    p2=np.array([point_top[1]+width+4 ,point_top[0]])
    back_letter=0
    return p2 , back_letter


def find_point_final_letter(consonant, image, width):
    l_pos_left = ['ฃ','ฑ','ซ','ธ','ฐ','ง','ว','ล','ว','ฉ','อ'] 
    l_pos_lefttop = ['ข','ฆ','ม','ช','ท','น','บ','ป','ผ','ฝ','พ','ฟ','ย','ษ','ห','ฬ']
    l_pos_leftbottom = ['ก','ค','ฅ','ฌ','ญ','ฎ','ฏ','ฒ','ณ','ด','ต','ถ','ภ','ศ','ส','ฤ']
    l_pos_bottomleft = ['ร']
    l_pos_mid = ['ฮ']

    if consonant in l_pos_left:
        p2 , back_letter = back_find_left(image, width)

    elif consonant in l_pos_lefttop:
        p2 , back_letter = back_find_left_top(image, width)
                                
    elif consonant in l_pos_leftbottom:
        p2 , back_letter = back_find_left_bottom(image, width)
                                
    elif consonant in l_pos_bottomleft:
        p2 , back_letter = back_find_bottom_left(image, width)
            
    else:
        p2 , back_letter = back_find_left_mid_from_bottom(image, width)

    return p2 , back_letter

def dot_line(image, dot, line):

    if dot:
        if line:
            black_pixel_positions = np.column_stack(np.where(image <= 10))
            #หาจุดต่ำสุด
            index_bottom=0
            for i, pos in enumerate(black_pixel_positions):
                if pos[0]>=index_bottom:
                    index_bottom=pos[0]
                    point_bottom=black_pixel_positions[i]
            #หาจุดขวาสุด
            index_right=0
            for i, pos in enumerate(black_pixel_positions):
                if pos[1]>=index_right and pos[0]>= point_bottom[0]-5:
                    index_right=pos[1]
                    point_bottom_right=black_pixel_positions[i]
            w = point_bottom_right[1]
            image = cv2.copyMakeBorder(image, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            point = (w,110)
            draw_line = draw_diagonal_line(image, point, 15)
            image = draw_line[0]
            line_end = draw_line[1]
            dot_point = [(line_end[0]+5,line_end[1]-5)]
            image = add_black_dots(image, dot_point, 3)

        else:
            black_pixel_positions = np.column_stack(np.where(image <= 10))
            #หาจุดต่ำสุด
            index_bottom=0
            for i, pos in enumerate(black_pixel_positions):
                if pos[0]>=index_bottom:
                    index_bottom=pos[0]
                    point_bottom=black_pixel_positions[i]
            #หาจุดขวาสุด
            index_right=0
            for i, pos in enumerate(black_pixel_positions):
                if pos[1]>=index_right and pos[0]>= point_bottom[0]-5:
                    index_right=pos[1]
                    point_bottom_right=black_pixel_positions[i]
            w = point_bottom_right[1]
            point = [(w+3, 99)]
            image = cv2.copyMakeBorder(image, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            image = add_black_dots(image,point, 3)
    else:
        if line:
            black_pixel_positions = np.column_stack(np.where(image <= 10))
            #หาจุดต่ำสุด
            index_bottom=0
            for i, pos in enumerate(black_pixel_positions):
                if pos[0]>=index_bottom:
                    index_bottom=pos[0]
                    point_bottom=black_pixel_positions[i]
            #หาจุดขวาสุด
            index_right=0
            for i, pos in enumerate(black_pixel_positions):
                if pos[1]>=index_right and pos[0]>= point_bottom[0]-5:
                    index_right=pos[1]
                    point_bottom_right=black_pixel_positions[i]
            w = point_bottom_right[1]
            image = cv2.copyMakeBorder(image, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            point = (w,110)
            draw_line = draw_diagonal_line(image, point, 15)
            image = draw_line[0]

    image = cv2.copyMakeBorder(image, 0, 0, 0, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return image

def if_tilt(image, tilt):
    if tilt:
        image = cv2.copyMakeBorder(image, 0, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        image = rotate_image(image,15)
    return image

def v_concat(name,style,nothing_omega_loop,tilt,dot,line):
    s_pos = ['ก','ข','ฃ','ค','ฅ','ฆ','ง','จ','ฉ','ช','ซ','ฌ','ญ','ฎ','ฏ','ฐ','ฑ','ฒ','ณ','ด','ต','ถ','ท','ธ','น','บ','ป','ผ','ฝ','พ','ฟ','ภ','ม','ย','ร','ล','ว','ศ','ษ','ส','ห','ฬ','อ','ฮ']
    cant_connect = ['ง','จ','ฉ','ฎ','ฏ','ล','ว','อ']
    s_pos_top = ['ข','ฃ','ฆ','ช','ซ','ฌ','ญ','ฐ','ฒ','ณ','ธ','น','บ','ป','ผ','ฝ','พ','ฟ','ม','ย','ร','ศ','ษ','ส','ฬ','ฮ']
    s_pos_bottom = ['ก','ค','ฅ','ฑ','ด','ต','ถ','ท','ภ','ห','ฤ']
    l_pos_left = ['ฃ','ฑ','ซ','ธ','ฐ','ง','ว','ล','ว','ฉ','อ'] 
    l_pos_lefttop = ['ข','ฆ','ม','ช','ท','น','บ','ป','ผ','ฝ','พ','ฟ','ย','ษ','ห','ฬ']
    l_pos_leftbottom = ['ก','ค','ฅ','ฌ','ญ','ฎ','ฏ','ฒ','ณ','ด','ต','ถ','ภ','ศ','ส','ฤ']
    l_pos_bottomleft = ['ร']
    l_pos_mid = ['ฮ']
    p0 = []
    p1 = []
    p2 = []
    consonant = []
    for i in name:
        if i in s_pos:
            consonant.append(i)
    if len(name)<1:
        return 
    if len(consonant)<2:
        image , consonant , omega , loop , head , h_ratio = genImage_for_one(name,style)
    else:
        image , consonant , omega , loop , head , h_ratio = genImage(name,style,nothing_omega_loop)
    print(consonant)
    #for i in image:
    #    plt.imshow(i, cmap="gray", interpolation="none")
    #    plt.axis("off")
    #    plt.show()
    padded_image = []
    for i in image:
        _, binary_image = cv2.threshold(i, 75, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = 0, 0, 0, 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter out small areas (noise)
                x, y, w, h = cv2.boundingRect(contour)
                print(x,y,w,h)
                break  # Stop at the first large contour, assuming it's the black line
                    
        cropped = i[y:y+h, x:x+w]
        padding_top = 100-h
        padded = cv2.copyMakeBorder(cropped, padding_top, 25, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        if padded is None:
            print("padded_imageไม่สามารถโหลดภาพได้")
        padded_image.append(padded)

    if len(consonant)<2:
        padded = dot_line(padded, dot, line)
        padded = if_tilt(padded, tilt)
        base64_string = image_to_base64(padded)
        result = {
        "image" : base64_string,
        "head_broken" : head[0],
        "head_cross" : head[1],
        "head_is" : consonant[0],
        "distance" : 0.5,
        "tall_ratio" : 0.5,
        "angle" : 0
        }
        return result
        
    
    if nothing_omega_loop == 0:
        for n in range(len(consonant)-1):
            if n==0:
                padded_image[n] = cv2.copyMakeBorder(padded_image[n], 0, 0, 0, 25, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                image_concat1 = cv2.hconcat([padded_image[n], padded_image[n+1]])
                if tilt:
                    ro_image = rotate_image(image_concat1, 15)
                    final_sig_v = final_value_contour_allSig(ro_image)
                    angle_error = 15 - final_sig_v[2]
                else:
                    final_sig_v = final_value_contour_allSig(image_concat1)
                    angle_error = 0 - final_sig_v[2]
            else:
                image_concat1 = cv2.copyMakeBorder(image_concat1, 0, 0, 0, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                image_concat1 = cv2.hconcat([image_concat1, padded_image[n+1]])
    else:
        for n in range(len(consonant)-1):
            if n==0:
                padded_image[n] = cv2.copyMakeBorder(padded_image[n], 0, 0, 0, 25, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                image_concat1 = cv2.hconcat([padded_image[n], padded_image[n+1]])
                if tilt:
                    ro_image = rotate_image(image_concat1, 15)
                    final_sig_v = final_value_contour_allSig(ro_image)
                    angle_error = 15 - final_sig_v[2]
                else:
                    final_sig_v = final_value_contour_allSig(image_concat1)
                    angle_error = 0 - final_sig_v[2]
                

            else:
                if len(consonant)>1:
                    if n==1 and consonant[1] in cant_connect:
                        image_concat1 = cv2.copyMakeBorder(image_concat1, 0, 0, 0, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                        image_concat1 = cv2.hconcat([image_concat1, padded_image[n+1]])
                    else:
                        height, width = image_concat1.shape
                        if n == 1:
                            if consonant[1] in s_pos_top:
                                p0 , front_letter = front_find_right_top(image_concat1)
                            else:
                                p0 , front_letter = front_find_right_bottom(image_concat1)
                                
                            if omega == 0 :
                                p2 , back_letter = back_find_left_top(padded_image[n+1], width)
                                p2[0]+=4

                            elif omega == 1:
                                p2 , back_letter = back_find_left_bottom(padded_image[n+1], width)
                                p2[0]+=4

                            elif loop == 0:
                                p2 , back_letter = back_find_top_left(padded_image[n+1], width)
                                p2[0]+=4
                                
                            elif loop == 1:
                                p2 , back_letter = back_find_bottom_left(padded_image[n+1], width)
                                p2[0]+=4
                                    
                            else:
                                p2 , back_letter = find_point_final_letter(consonant[-1], padded_image[n+1], width)
                                p2[0]+=4

                        else:
                            if omega == 0 or loop == 0:
                                p0 , front_letter = front_find_right_top(image_concat1)
                                    
                            elif omega == 1 or loop == 1:
                                p0 , front_letter = front_find_right_bottom(image_concat1)
                                
                            if n != len(consonant)-2:
                                if omega == 0 :
                                    p2 , back_letter = back_find_left_top(padded_image[n+1], width)
        
                                elif omega == 1:
                                    p2 , back_letter = back_find_left_bottom(padded_image[n+1], width)
                                
                                elif loop == 0:
                                    p2 , back_letter = back_find_top_left(padded_image[n+1], width)

                                else:
                                    p2 , back_letter = back_find_bottom_left(padded_image[n+1], width)
                                    
                            else:
                                p2 , back_letter = find_point_final_letter(consonant[-1], padded_image[n+1], width)
                                p2[0]+=4
                        
                        if n == 1 or n == len(consonant)-2:
                            image_concat1 = cv2.copyMakeBorder(image_concat1, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                            image_concat1 = cv2.hconcat([image_concat1, padded_image[n+1]])
                        else:
                            image_concat1 = cv2.copyMakeBorder(image_concat1, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                            image_concat1 = cv2.hconcat([image_concat1, padded_image[n+1]])

                    
                        if front_letter == back_letter:
                            if front_letter==0:
                                p1=np.array([p0[0]+2 ,p0[1]-5])
                            else:
                                p1=np.array([p0[0]+2 ,p0[1]+5])
            
                        else:
                            if front_letter==0:
                                p1=np.array([p0[0] ,p2[1]])
                            else:
                                p1=np.array([p0[0] ,p2[1]])
                        print(f'รอบ{n} {p0},{p1},{p2}')        
                        curve_points = []
                        for t in np.linspace(0, 1, 5):  # t จาก 0 ถึง 1
                            point = quadratic_bezier(t, p0, p1, p2)
                            curve_points.append(point)
                        curve_points = np.array(curve_points, dtype=np.int32)
                        for i in range(len(curve_points) - 1):
                            cv2.line(image_concat1, curve_points[i], curve_points[i+1], (0, 255, 0), 2, cv2.LINE_AA)
    image_concat1 = dot_line(image_concat1, dot, line)
    image_concat1 = if_tilt(image_concat1, tilt)
    
    blur = cv2.GaussianBlur(image_concat1, (3,3), 0, borderType=cv2.BORDER_REPLICATE)
    base64_string = image_to_base64(image_concat1)
    result = {
        "image" : base64_string,
        "head_broken" : head[0],
        "head_cross" : head[1],
        "head_is" : consonant[0],
        "distance" : final_sig_v[0],
        "tall_ratio" : h_ratio,
        "angle" : angle_error
    }
    return result

