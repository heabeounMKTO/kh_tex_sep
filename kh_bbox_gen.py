'''
generate khmer text clusters where each cluster has a max of ONE (1) coeng sequence 

usage example:
```python

from kh_bbox_gen import KhmerTextClusterGenerator
renderer = KhmerTextRenderer(font_path="./Hanuman-Regular.ttf", font_size=24)
test_text = "សានកែវមនោរា"
clusters = renderer.render_text(test_text, "khmer_text_with_boxes.png")
```

'''

from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import os 
import unicodedata
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union



class KhmerTextClusterGenerator:
    def __init__(self, font_path="Khmer.ttf", font_size=48, background_images_dir: str = "random_images"):
        self.font = ImageFont.truetype(font_path, font_size)
        self.base_height = font_size
        self.background_images_dir = background_images_dir
        # Get standard character height using a tall character like ក
        self.standard_height = (self.font.getbbox('ក')[3] - self.font.getbbox('ក')[1]) * 2
        
    def analyze_khmer_cluster(self, text):
        """
        Analyze Khmer text to identify character clusters and their components.
        A cluster includes a base character and all its dependent vowels, signs,
        and subscript consonants.
        """
        clusters = []
        current_cluster = []
        
        i = 0
        while i < len(text):
            char = text[i]
            if char == '\u17AB':
                # If there's an existing cluster, save it
                if current_cluster:
                    clusters.append(current_cluster)
                # Create a new cluster with just this character
                clusters.append([char])
                current_cluster = []
                i += 1
                continue 

            # start a new cluster if we're at a base consonant
            if not current_cluster or self.is_base_consonant(char):
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [char]
            else:
                # add character to current cluster if it's a dependent sign
                current_cluster.append(char)
            
            # special handling for coeng (្) sequences
            if i + 1 < len(text) and text[i + 1] == '\u17D2':  # COENG
                # Include COENG and the following consonant in the current cluster
                current_cluster.extend([text[i + 1], text[i + 2]])
                i += 2  # Skip the next two characters as we've already processed them
            i += 1
        
        # Add the last cluster if there is one
        if current_cluster:
            clusters.append(current_cluster)
            
        return clusters
    
    def is_base_consonant(self, char):
        """
        Check if a character is a Khmer base consonant.
        """
        # Khmer consonant range
        return '\u1780' <= char <= '\u17A2'
    
    def is_vowel_sign(self, char):
        """
        Check if a character is a Khmer vowel sign.
        """
        # Khmer vowel sign range
        return '\u17B6' <= char <= '\u17C5'

    def calculate_cluster_height(self, cluster):
        """
        Calculate the height for a cluster, including all adjustments.
        """
        cluster_text = ''.join(cluster)
        bbox = self.font.getbbox(cluster_text)
        
        # Start with standard height
        height = max(self.standard_height, bbox[3] - bbox[1])
        
        # Adjust for clusters with COENG
        if '\u17D2' in cluster_text:
            height += self.base_height * 0.3
        
        # Adjust for vowel signs
        for char in cluster:
            if self.is_vowel_sign(char):
                height += self.base_height * 0.1
                
        return height

    def get_char_bbox(self, cluster, x, y, max_height):
        """
        Get the bounding box for a character cluster.
        All boxes will have the same height as the tallest cluster.
        """
        cluster_text = ''.join(cluster)
        bbox = self.font.getbbox(cluster_text)
        
        # Base width from font metrics
        width = bbox[2] - bbox[0]
        
        return (x + bbox[0], y, x + bbox[2], y + max_height)
            
    def add_noise(self, image: Image.Image, noise_factor: float = 0.1) -> Image.Image:
        """Add random noise to the image"""
        img_array = np.array(image)
        noise = np.random.normal(0, noise_factor * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def apply_emboss(self, image: Image.Image, probability: float = 0.5):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        apply = random.random() < probability
        if apply:
            return image.filter(ImageFilter.EMBOSS)
        else:
            return image 
    
    def apply_invert(self, image: Image.Image, probability: float = 0.5):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        apply = random.random() < probability
        if apply:
            return ImageOps.invert(image) 
        else:
            return image 

    def apply_random_blur(self, image: Image.Image, max_radius: float = 1.0) -> Image.Image:
        """Apply Gaussian blur with random radius"""
        radius = random.uniform(0, max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))

    def adjust_brightness_contrast(self, image: Image.Image) -> Image.Image:
        """Randomly adjust brightness and contrast"""
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_factor)

    def get_random_background(self, size: Tuple[int, int]) -> Optional[Image.Image]:
        """Get a random background image from the specified directory"""
        if not self.background_images_dir or not os.path.exists(self.background_images_dir):
            return None
            
        bg_files = [f for f in os.listdir(self.background_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not bg_files:
            return None
            
        bg_file = random.choice(bg_files)
        bg_image = Image.open(os.path.join(self.background_images_dir, bg_file))
        return bg_image.resize(size, Image.Resampling.LANCZOS)


    def render_text(self, text,  output_path: str, save_img: bool = False, augment: bool = True, augmentation_params = None):
        """
        Render Khmer text and draw bounding boxes for each character cluster.
        All bounding boxes will have the same height as the tallest cluster.
        
        Returns:
            List[Dict]: List of clusters with their text and bounding boxes
        """
        # Create image with white background
        default_params = {
            'noise_factor': 0.1,
            'max_blur': 1.0,
            'use_background': False,
            'rotation_range': (-5, 5),
            'emboss': 0.0,
            'invert': 0.5
        }
        augmentation_params = {**default_params, **(augmentation_params or {})}
        
        # Create image with white background
        padding = 50
        img_width = self.font.getlength(text) + padding * 2
        img_height = self.base_height * 2 + padding * 2
        
        # Get background image if requested
        if augment and augmentation_params['use_background']:
            base_image = self.get_random_background((int(img_width), int(img_height)))
            if base_image is None:
                base_image = Image.new('RGB', (int(img_width), int(img_height)), 'white')
        else:
            base_image = Image.new('RGB', (int(img_width), int(img_height)), 'white')
            
        # Create transparent layer for text
        text_layer = Image.new('RGBA', (int(img_width), int(img_height)), (255, 255, 255, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # Analyze text into clusters
        clusters = self.analyze_khmer_cluster(text)
        max_height = max(self.calculate_cluster_height(cluster) for cluster in clusters)
        
        # Current x position for drawing
        x = padding
        y = padding + self.base_height * 0.3
        
        # List to store cluster information
        cluster_info = []
        
        # Draw text and bounding boxes
        for cluster in clusters:
            cluster_text = ''.join(cluster)
            draw.text((x, y), cluster_text, font=self.font, fill='black')
            
            bbox = self.get_char_bbox(cluster, x, y, max_height)
            if save_img:
                draw.rectangle(bbox, outline='red', width=1)
            
            cluster_info.append({
                'text': cluster_text,
                'bbox': bbox,
                'components': cluster,
                'is_complex': len(cluster) > 1 or '\u17D2' in cluster_text
            })
            
            x += self.font.getlength(cluster_text)
        
        # Composite text layer onto background
        result_image = Image.alpha_composite(base_image.convert('RGBA'), text_layer)
        result_image = result_image.convert('RGB')
        
        # Apply augmentations if requested
        if augment:
            # Random rotation
            if augmentation_params['rotation_range'] != (0, 0):
                angle = random.uniform(*augmentation_params['rotation_range'])
                result_image = result_image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
            
            if augmentation_params["invert"] > 0:
                result_image = self.apply_invert(result_image, augmentation_params["invert"])
            if augmentation_params['noise_factor'] > 0:
                result_image = self.add_noise(result_image, augmentation_params['noise_factor'])
            if augmentation_params['emboss'] > 0:
                result_image = self.apply_emboss(result_image, augmentation_params["emboss"])
            if augmentation_params['max_blur'] > 0:
                result_image = self.apply_random_blur(result_image, augmentation_params['max_blur'])
            result_image = self.adjust_brightness_contrast(result_image)
        
        # Save the image
        if save_img:
            result_image.save(output_path)
            
        return {
            "cluster_info": cluster_info,
            "rendered_image": result_image
        }
