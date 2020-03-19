#!/usr/bin/env python3

import math
import sys
import math
import base64
import tkinter
import random

from io import BytesIO
from PIL import Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, x, y):
    if x < 0:
        x = 0
    elif x > image['width'] - 1:
        x = image['width'] - 1
    if y < 0:
        y = 0
    elif y > image['height'] - 1:
        y = image['height'] - 1
    return image['pixels'][y * image['width'] + x] # list index has to be int or slice


def set_pixel(image, x, y, c):
    image['pixels'][y * image['width'] + x] = c # list index has to be int or slice


def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [None] * (image['height'] * image['width']), # initialize w x h size array
    }
    for x in range(image['width']):
        for y in range(image['height']):
            color = get_pixel(image, x , y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor) # y, x to x, y and indentation
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c) # 256 to 255


# HELPER FUNCTIONS

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    The kernel is a square with an odd number of rows and columns and
    is represented as a 1-D list containing n*n elements.
    """
    
    # Initialize new image    
    new_image = {
            'height': image['height'], 
            'width': image['width'], 
            'pixels': [i for i in image['pixels']]
                }
    
    # Modify pixels (x corresponds to height/rows, y corresponds to width/columns)
    for x in range(image['width']):
        for y in range(image['height']):
            
            # Get size of kernel
            n = int(math.sqrt(len(kernel)))
            image_portion = []
            half_length = n // 2
            for i in range(-half_length, half_length + 1):
                for j in range(-half_length, half_length + 1):
                    image_portion.append(get_pixel(image, x + j, y + i))
                    
            # Compute correlate values for each pixel
            correlate_values = [a*b for a,b in zip(image_portion, kernel)]          
            new_pixel_value = sum(correlate_values)
            set_pixel(new_image, x, y, new_pixel_value)
            
    return new_image
  

def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    
    new_pixels = []
    
    # Clip out of range values
    for p in image['pixels']:
        if p > 255:
            p = 255   
        elif p < 0:
            p = 0
        
        # Round final values
        new_pixels.append(round(p))
        
    return {'height': image['height'], 
            'width': image['width'], 
            'pixels': new_pixels}
        

def box_blur_kernel(n):
    return [1 / (n**2)] * n**2

# FILTERS

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = box_blur_kernel(n)

    # then compute the correlation of the input image with that kernel
    blurred_image = correlate(image, kernel) # shouldn't mutate because i copy in correlate, right?

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    return round_and_clip_image(blurred_image)


def sharpened(i, n):
    """
    Return a new image representing the result of applying 2I−B 
    (where I is the original pixel value and B is the blurred version of it), 
    with kernel size n, to the given input image.
    """
    # Blur the image, but do not round/clip values
    kernel = box_blur_kernel(n)
    blurred_i = correlate(i, kernel)
    
    # Apply formula to each pixel
    sharpened_color_list = []
    for p in range(len(i['pixels'])):
        color = i['pixels'][p]
        sharpened_color = (2 * color) - blurred_i['pixels'][p]
        sharpened_color_list.append(sharpened_color)
    result = {'height': i['height'], 
            'width': i['width'], 
            'pixels': sharpened_color_list}
    
    # Round and clip 
    return round_and_clip_image(result)


def edges(i):
    Kx = [-1, 0, 1,
          -2, 0, 2,
          -1, 0, 1]
    Ky = [-1, -2, -1,
          0, 0, 0,
          1, 2, 1]
    i_Kx = correlate(i, Kx)
    i_Ky = correlate(i, Ky)
    
    # Apply formula to each pixel
    new_color_list = []
    for p in range(len(i['pixels'])):
        new_color = round(math.sqrt((i_Kx['pixels'][p] ** 2) + (i_Ky['pixels'][p] ** 2)))
        new_color_list.append(new_color)
        
    result = {'height': i['height'], 
            'width': i['width'], 
            'pixels': new_color_list}
    
    # Round and clip 
    return round_and_clip_image(result)
    

# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()



# VARIOUS FILTERS

def get_r(image):
    """
    Given an image as input, returns the red values for each pixel 
    as a list of integers [0, 255].
    """
    return [p[0] for p in image['pixels']]

def get_g(image):
    """
    Given an image as input, returns the green values for each pixel 
    as a list of integers [0, 255].
    """
    return [p[1] for p in image['pixels']]

def get_b(image):
    """
    Given an image as input, returns the blue values for each pixel 
    as a list of integers [0, 255].
    """
    return [p[2] for p in image['pixels']]


def image_with_new_pix(image, pix):
    """
    Given an image and a list of updated pixel values as input, 
    returns a new image with same size and updated pixel values.
    """    
    
    return {'height': image['height'], 
            'width': image['width'], 
            'pixels': pix}

def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def color_filter(i):
        # Split the color image into its three components r, g, and b
        # Apply the greyscale filter to each color component 
        r_filtered = filt(image_with_new_pix(i, get_r(i)))
        g_filtered = filt(image_with_new_pix(i, get_g(i)))
        b_filtered = filt(image_with_new_pix(i, get_b(i)))

        return image_with_new_pix(i, list(zip(r_filtered['pixels'], g_filtered['pixels'], b_filtered['pixels'])))
    
        
    return color_filter
    

def make_blur_filter(n):
    def blur_filter(i):
        return blurred(i, n)
    return blur_filter


def make_sharpen_filter(n):
    def sharpen_filter(i):
        return sharpened(i, n)
    return sharpen_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    
    def apply_all_filters(i):
        # Apply first filter
        new_i = filters[0](i)
        # Return new image after applying all the other filters
        for f in filters[1:]:
            new_i = f(new_i)   
        return new_i
    
    return apply_all_filters
            
        

# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """

    
    # Remove seams
    for i in range(ncols):
        # Make a greyscale copy
        grey = greyscale_image_from_color_image(image)
    
        # Compute energy map
        energy_map = compute_energy(grey)
    
        # Compute a cumulative energy map
        c_map = cumulative_energy_map(energy_map)
        s = minimum_energy_seam(c_map)
        image = image_without_seam(image, s)
    return image
    
 
# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    # Get r, g, b values for each pixel
    num_pixels = image['height'] * image['width']
    r, g, b = get_r(image), get_g(image), get_b(image)
    gray_pixels = [round((r[i] * .299) + (g[i] * .587) + (b[i] * .114)) 
                    for i in range(num_pixels)]
    
    return image_with_new_pix(image, gray_pixels)
    

def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    
    return edges(grey) 


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy function),
    computes a "cumulative energy map" as described in the lab 1 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    
    # For each pixel in energy, starting from second row ...
    for y in range(1, energy['height']): # row
        for x in range(energy['width']): # column
            
            # Get adjacent pixel values
            top_left_pix = get_pixel(energy, x - 1, y -1)
            top_pix = get_pixel(energy, x , y -1)
            top_right_pix = get_pixel(energy, x + 1 , y -1)
            min_adj = min(top_left_pix, top_pix, top_right_pix)
            
            # Compute new pixel value, update energy
            new_pixel_value = get_pixel(energy, x, y) + min_adj
            set_pixel(energy, x, y, new_pixel_value)
            
    return energy


def minimum_energy_seam(c): 
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 1 writeup).
    """
    
    index = []
    
    # Get index of min pixel in bottom row
    bottom_row_index = (len(c['pixels']) - c['width']) 
    bottom_row_min = min(c['pixels'][bottom_row_index : ])
    bottom_row_x = c['pixels'][bottom_row_index : ].index(bottom_row_min)
    
    # Calculate and keep index of bottom row left-most min pixel value 
    bottom_row_min_index = bottom_row_index + bottom_row_x 
    index.append(bottom_row_min_index)
           
    # Starting from bottom row going up
    for y in range(c['height'] - 1, 0, -1): # row
        
        # If pixel is on the left edge
        if bottom_row_x == 0:
            top_pix = get_pixel(c, bottom_row_x , y -1)
            top_right_pix = get_pixel(c, bottom_row_x + 1 , y -1)
            min_adj_list = [top_pix, top_right_pix]
            min_adj = min_adj_list.index(min(min_adj_list))
            
            bottom_row_x = bottom_row_x + min_adj

        # If pixel is on the right edge
        elif bottom_row_x == c['width'] - 1:
            top_left_pix = get_pixel(c, bottom_row_x - 1, y -1)
            top_pix = get_pixel(c, bottom_row_x , y -1)
            min_adj_list = [top_left_pix, top_pix]
            min_adj = min_adj_list.index(min(min_adj_list))

            bottom_row_x = bottom_row_x + min_adj -1

        # For non-edge pixels
        else:
            top_left_pix = get_pixel(c, bottom_row_x - 1, y -1)
            top_pix = get_pixel(c, bottom_row_x , y -1)
            top_right_pix = get_pixel(c, bottom_row_x + 1 , y -1)  
            min_adj_list = [top_left_pix, top_pix, top_right_pix]
            min_adj = min_adj_list.index(min(min_adj_list))

            bottom_row_x = bottom_row_x + min_adj -1
            
        min_adj_index = c['width'] * (y -1) + bottom_row_x
        index.append(min_adj_index)
            

    index.reverse()
    return index

  
def image_without_seam(im, s):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    # Copy pixels
    im_pixels = [p for p in im['pixels']]
    pixels = []
    
    # Keep pixel values with indices not in s
    for i in range(len(im_pixels)):
        if i not in s:
            pixels.append(im_pixels[i])
            
    # Return new re-sized image
    return {'height': im['height'], 
            'width': im['width'] - 1, 
            'pixels': pixels}

# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()

# HELPER FUNCTIONS FOR THE CREATIVE IMPLEMENTATION
    
def random_kernel(n):
    return [(random.random() / 2)] * n**2


def random_filter(i):
    """
    Return a new image representing the result of applying a
    kernel size n with randomly generated values, to the given input image.
    """
    # Create and apply random kernel
    kernel = random_kernel(1)
    random_i = correlate(i, kernel)
    
    # Round and clip 
    return round_and_clip_image(random_i)

    
    
#def low_energy_kernel(adj_pix):
#    """
#    Given a list of adjacent pixels (3 top pixel values), creates a 
#    location-specific kernel for computing cumulative energy map for the pixel.
#
#    Returns a 3x3 kernel.
#    """
#    # Get index of adjacent pixel with lowest value
#    min_adj_index = adj_pix.index(min(adj_pix)) # returns the lowest index (left-most pixel if there's a tie)
#   
#    # Initialize kernel
#    low_energy_kernel = [0, 0, 0,
#                         0, 1, 0,
#                         0, 0, 0]
#    
#    # Update kernel with low energy adjacent pixel
#    low_energy_kernel[min_adj_index] = 1
#    return low_energy_kernel


#def correlate_pixel(image, pixel_x, pixel_y, kernel):
#    """
#    Compute the result of correlating ONE pixel with the given kernel.
#
#    The kernel is a square with an odd number of rows and columns and
#    is represented as a 1-D list containing n*n elements.
#    """
#    
#    # Get size of kernel
#    n = int(math.sqrt(len(kernel)))
#    image_portion = []
#    half_length = n // 2
#    
#    for i in range(-half_length, half_length + 1):
#        for j in range(-half_length, half_length + 1):
#            image_portion.append(get_pixel(image, pixel_x + j, pixel_y + i))
#            
#    # Compute correlate values for pixel
#    correlate_values = [a*b for a,b in zip(image_portion, kernel)]         
#    new_pixel_value = sum(correlate_values)
#    
#    return new_pixel_value

    
#def cumulative_energy_map(energy):
#    """
#    Given a measure of energy (e.g., the output of the compute_energy function),
#    computes a "cumulative energy map" as described in the lab 1 writeup.
#
#    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
#    the values in the 'pixels' array may not necessarily be in the range [0,
#    255].
#    """
#
#    # For each pixel in energy...
#    for y in range(1, energy['height']):
#        for x in range(energy['width']):
#            
#            # Get adjacent pixel values
#            top_left_pix = get_pixel(energy, x - 1, y -1)
#            top_pix = get_pixel(energy, x , y -1)
#            top_right_pix = get_pixel(energy, x + 1 , y -1)
##            print("x,y" ,x, y)
##            print("adj pixels:", top_left_pix, top_pix, top_right_pix)
#            
#            # Make pixel-specific low energy kernel
#            kernel = low_energy_kernel([top_left_pix, top_pix, top_right_pix]) 
#            
#            # Compute new pixel value, update energy
#            new_pixel_value = correlate_pixel(energy, x, y, kernel)
#            set_pixel(energy, x, y, new_pixel_value)
#            
#    return energy
 
    
if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    pass

    #### 4.1 Test: Testing the inversion filter ####
#    i = load_color_image('test_images/cat.png')
#    inversion_filter = color_filter_from_greyscale_filter(inverted) 
#    inverted_i = inversion_filter(i)
#    new_i = save_color_image(inverted_i, 'test_images/inverted_color_cat.png', mode='PNG')
    
  #######################################################

    #### 4.3 Test 1: Testing make_blur_filter with n=9 ####
#    n = 9
#    i = load_color_image('test_images/python.png')
#    blur_filter = color_filter_from_greyscale_filter(make_blur_filter(n)) 
#    blurred_i = blur_filter(i)
#    new_i = save_color_image(blurred_i, 'test_images/blurred_python.png', mode='PNG')
    
  #######################################################

    #### 4.3 Test 2: Testing make_sharpen_filter with n=7 ####
    
#    n = 7
#    i = load_color_image('test_images/sparrowchick.png')
#    sharpen_filter = color_filter_from_greyscale_filter(make_sharpen_filter(n)) 
#    sharpened_i = sharpen_filter(i)
#    new_i = save_color_image(sharpened_i, 'test_images/sharpened_sparrowchick.png', mode='PNG')
    
  #######################################################

    #### 5.1 Test : Testing filter_cascade ####
    
#    filter1 = color_filter_from_greyscale_filter(edges)
#    filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
#    filt = filter_cascade([filter1, filter1, filter2, filter1])
#    i = load_color_image('test_images/frog.png')
#    filtered_i = filt(i)
#    new_i = save_color_image(filtered_i,'test_images/filtered_frog.png', mode='PNG')
    
  #######################################################

    #### *6.1 Test : Testing greyscale_image_from_color_image ####
    
#    i = load_color_image('test_images/cat.png')
#    greyscaled_i = greyscale_image_from_color_image(i) 
#    new_i = save_greyscale_image(greyscaled_i, 'test_images/greyscaled_cat.png')
    
  #######################################################

    #### 6.5 Test : Testing seam_carving by removing 100 seams ####
    
#    n = 100
#    i = load_color_image('test_images/twocats.png')    
#    smaller_i = seam_carving(i, n)
#    new_i = save_color_image(smaller_i,'test_images/smaller_twocats.png', mode='PNG')
    
  #######################################################

    #### 7 Test : Something of your own ####
    
    i = load_greyscale_image('test_images/greyscaled_cat.png') 
    filt = filter_cascade([edges, random_filter, edges, random_filter])
    filtered_i = filt(i)
    new_i = save_greyscale_image(filtered_i,'test_images/random_greyscaled_cat4.png', mode='PNG')    
    
    print("Findings: The larger the matrix, the more local information is lost.")
  #######################################################

    
#    i = load_color_image('test_images/cat.png')
#    cum_map_i = round_and_clip_image(cumulative_energy_map(compute_energy(greyscale_image_from_color_image(i))))
#    new_i = save_greyscale_image(cum_map_i, 'test_images/cum_map_cat.png')
    
#    i = {'width': 3, 'height': 3, 'pixels': [(160, 87, 90), (5, 6, 9), (10, 97, 40), 
#                                             (53, 0, 4), (160, 87, 90), (5, 6, 9),
#                                             (160, 87, 90), (5, 6, 9), (10, 97, 40)]}
#    i = load_color_image('test_images/pattern.png')
#    compute_i = cumulative_energy_map(compute_energy(greyscale_image_from_color_image(i))) 
##    print(compute_i)
##    compute_answer = {'width': 9, 'height': 4, 'pixels': [160, 160, 0, 28, 0, 28, 0, 160, 160, 415, 218, 10, 22, 14, 22, 10, 218, 415, 473, 265, 40, 10, 28, 10, 40, 265, 473, 520, 295, 41, 32, 10, 32, 41, 295, 520]}
#    i_energy_seam = minimum_energy_seam(compute_i)
#    i_energy_seam_answer = [2, 11, 21, 31]
#    print("result seam:", i_energy_seam)
#    print("expected seam:", i_energy_seam_answer)
#    print(compute_i)
#    print(compute_i == compute_answer)
#    print(i_energy_seam)
#    print(i_energy_seam == i_energy_seam_answer)
    
    
       