import argparse
import time
import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models_x import *
from datasets_evaluation import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/home/liangjie/AIPS_data/", help="root of the datasets")
parser.add_argument("--gpu_id", type=str, default="7", help="gpu id")
parser.add_argument("--epoch", type=int, default=-1, help="epoch to load")
parser.add_argument("--model_dir", type=str, default="mask_noglc_a", help="path to save model")
parser.add_argument("--lut_dim", type=int, default=33, help="dimension of lut")
opt = parser.parse_args()
print(f"Attempting to load images from data_path: '{opt.data_path}'")
absolute_data_path = os.path.abspath(opt.data_path)
print(f"Absolute path resolves to: '{absolute_data_path}'")

# Check if the directory even exists
if not os.path.isdir(absolute_data_path):
    print("!!! ERROR: This directory does not exist. Please check your --data_path argument.")
else:
    print("Directory exists. Proceeding to create dataset.")


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

#cuda = True if torch.cuda.is_available() else False
cuda = False

criterion_pixelwise = torch.nn.MSELoss()
# Initialize generator and discriminator
LUT1 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT2 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT3 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT4 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT5 = Generator3DLUT_identity(dim=opt.lut_dim)
classifier = resnet18_224(out_dim=5)
trilinear_ = TrilinearInterpolation()

if cuda:
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    LUT3 = LUT3.cuda()
    LUT4 = LUT4.cuda()
    LUT5 = LUT5.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.model_dir, opt.epoch))
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
LUT3.load_state_dict(LUTs["3"])
LUT4.load_state_dict(LUTs["4"])
LUT5.load_state_dict(LUTs["5"])

classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.model_dir, opt.epoch)))
classifier.eval()

# Create the dataset first so we can check its length
dataset = ImageDataset_paper(opt.data_path)
print(f"--- The ImageDataset_paper found {len(dataset)} images. ---")

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def generator(img):
    print("\n--- INSIDE GENERATOR FUNCTION ---")

     # [Debug 1/4] Check the input tensor
    print(f"[Generator Debug 1/4] Input 'img' tensor properties:")
    print(f"  > Shape: {img.shape}")
    print(f"  > Dtype: {img.dtype}")
    # Check if the tensor has valid numbers and what its range is
    if torch.is_tensor(img) and img.numel() > 0:
        print(f"  > Min value: {torch.min(img)}")
        print(f"  > Max value: {torch.max(img)}")
        print(f"  > Contains NaNs: {torch.isnan(img).any()}")
    print("-" * 20)


    pred = classifier(img).squeeze()

    # [Debug 2/4] Check the classifier's output
    print(f"[Generator Debug 2/4] 'pred' tensor (output of classifier):")
    print(f"  > Shape: {pred.shape}")
    print(f"  > Dtype: {pred.dtype}")
    if torch.is_tensor(pred) and pred.numel() > 0:
        print(f"  > Min value: {torch.min(pred)}")
        print(f"  > Max value: {torch.max(pred)}")
        print(f"  > Contains NaNs: {torch.isnan(pred).any()}")
    print("-" * 20)


    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)

    # --- START OF MEMORY-INTENSIVE OPERATIONS ---
    try:
        # We will now monitor each LUT call
        print("[Generator Debug 3/5] Applying LUTs...")

        print("  > Applying LUT1...")
        gen_A1 = LUT1(img)
        print("    - LUT1 finished.")

        print("  > Applying LUT2...")
        gen_A2 = LUT2(img)
        print("    - LUT2 finished.")

        print("  > Applying LUT3...")
        gen_A3 = LUT3(img)
        print("    - LUT3 finished.")

        print("  > Applying LUT4...")
        gen_A4 = LUT4(img)
        print("    - LUT4 finished.")

        print("  > Applying LUT5...")
        gen_A5 = LUT5(img)
        print("    - LUT5 finished.")

        print("[Generator Debug 4/5] Combining results...")
        combine_A = img.new(img.size())
        for i in range(img.size(0)):
            combine_A[i, :, :, :] = (
                        pred[i, 0] * gen_A1[i, :, :, :] + pred[i, 1] * gen_A2[i, :, :, :] + pred[i, 2] * gen_A3[i, :, :,
                                                                                                        :] +
                        pred[i, 3] * gen_A4[i, :, :, :] + pred[i, 4] * gen_A5[i, :, :, :])
        print("  > Combination finished successfully.")

        print("[Generator Debug 5/5] Generator function finished successfully.")
        return combine_A

    except RuntimeError as e:
        # This block will ONLY run if an error occurs inside the 'try' block
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CAUGHT A RUNTIME ERROR! This is the crash point.")
        print(f"!!! Error Message: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        # To prevent further crashes, we return the original image
        return img
    # --- END OF MEMORY-INTENSIVE OPERATIONS ---

def visualize_result():
    """Saves a generated sample from the validation set"""
    out_dir = "results/%s_%d" % (opt.model_dir, opt.epoch)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nStarting visualization loop. Will save results to: '{os.path.abspath(out_dir)}'")
    if len(dataloader) == 0:
        print("!!! WARNING: DataLoader is empty. The loop will not run. No images will be saved. !!!")

    for i, batch in enumerate(dataloader):
        print(f"Processing batch {i+1}/{len(dataloader)}...") # This will tell us if the loop is running
        real_A = Variable(batch["A_input"].type(Tensor))
        img_name = batch["input_name"]
        fake_B = generator(real_A)

        # Assume fake_B is BGR and swap it to RGB for saving.
        # The tensor shape is (Batch, Channel, Height, Width), so we swap the Channel dimension (dim=1).
        # fake_B_rgb = fake_B.clone() # Create a copy to avoid modifying the original
        # fake_B_rgb = fake_B_rgb[:, [2, 1, 0], :, :] # Swaps Red and Blue channels

        output_path = os.path.join(out_dir, "%s.png" % (img_name[0][:-4]))
        save_image(fake_B, output_path, nrow=1, normalize=False)
        print(f"  > Saved image to {output_path}")

    print("--- Finished visualize_result function. ---")

# --- THE FIX ---
# Put your main execution logic inside this block
if __name__ == '__main__':
    # This code will now ONLY run when you execute "python validation.py"
    # It will NOT run when a child process imports this file.
    visualize_result()
