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

from PIL import Image,ImageFont, ImageDraw
import os 
import unicodedata
from typing import List, Dict, Tuple

class KhmerTextClusterGenerator:
    def __init__(self, font_path="Khmer.ttf", font_size=48):
        self.font = ImageFont.truetype(font_path, font_size)
        self.base_height = font_size
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
            
    def render_text(self, text,  output_path: str, save_img: bool = False):
        """
        Render Khmer text and draw bounding boxes for each character cluster.
        All bounding boxes will have the same height as the tallest cluster.
        
        Returns:
            List[Dict]: List of clusters with their text and bounding boxes
        """
        # Create image with white background
        padding = 50
        img_width = self.font.getlength(text) + padding * 2
        img_height = self.base_height * 2 + padding * 2
        image = Image.new('RGB', (int(img_width), int(img_height)), 'white')
        draw = ImageDraw.Draw(image)
        
        # Analyze text into clusters
        clusters = self.analyze_khmer_cluster(text)
        
        # Calculate maximum height among all clusters
        max_height = max(self.calculate_cluster_height(cluster) for cluster in clusters)
        
        # Current x position for drawing
        x = padding
        y = padding + self.base_height * 0.3  # Adjust baseline position
        
        # List to store cluster information
        cluster_info = []
        
        # Draw text and bounding boxes
        for cluster in clusters:
            cluster_text = ''.join(cluster)
            
            # Draw the text
            draw.text((x, y), cluster_text, font=self.font, fill='black')
            
            # Get and draw bounding box with normalized height
            bbox = self.get_char_bbox(cluster, x, y, max_height)
            if save_img:
                draw.rectangle(bbox, outline='red', width=1)
            
            # Store cluster information
            cluster_info.append({
                'text': cluster_text,
                'bbox': bbox,
                'components': cluster,
                'is_complex': len(cluster) > 1 or '\u17D2' in cluster_text
            })
            
            # Move x position by cluster width
            x += self.font.getlength(cluster_text)
        
        # Save the image
        if save_img:
            image.save(output_path)
        return {"cluster_info": cluster_info, "rendered_image": image}
