import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def bandpass_mask(shape, low_radius, high_radius):
    """Create a 2D radial band-pass mask (centered)."""
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    y = np.arange(0, rows) - crow
    x = np.arange(0, cols) - ccol
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    mask = np.logical_and(D >= low_radius, D <= high_radius).astype(np.float32)
    return mask

def highpass_mask(shape, cutoff_radius, order=2):
    """Create a 2D high-pass mask (Butterworth-style) to emphasize edges/structures."""
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    y = np.arange(0, rows) - crow
    x = np.arange(0, cols) - ccol
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    # Butterworth high-pass: smooth transition, emphasizes mid-to-high frequencies
    mask = 1.0 / (1.0 + (cutoff_radius / (D + 1e-10)) ** (2 * order))
    return mask.astype(np.float32)

def weighted_bandpass_mask(shape, low_radius, high_radius, smoke_emphasis=True):
    """Create a bandpass mask with emphasis on smoke frequencies (mid-range)."""
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    y = np.arange(0, rows) - crow
    x = np.arange(0, cols) - ccol
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    
    # Base bandpass
    base_mask = np.logical_and(D >= low_radius, D <= high_radius).astype(np.float32)
    
    if smoke_emphasis:
        # Weight mid-frequencies more heavily (where smoke patterns live)
        mid_freq = (low_radius + high_radius) / 2
        # Gaussian weighting centered at mid-frequencies
        weight = np.exp(-0.5 * ((D - mid_freq) / (high_radius - low_radius)) ** 2)
        # Combine: keep bandpass region, but weight mid-frequencies more
        mask = base_mask * (0.5 + 0.5 * weight)
    else:
        mask = base_mask
    
    return mask.astype(np.float32)

def enhance_image_bandpass(img, low_radius=10, high_radius=60, keep_color=False, blend_factor=0.3, 
                           filter_type='bandpass', smoke_emphasis=True, save_spectrum=False, spectrum_path=None):
    """
    img: BGR uint8 image
    low_radius, high_radius: radii in frequency-domain pixels (tuneable)
    keep_color: if True, apply same filter to each channel and recombine
    blend_factor: how much of the enhanced features to blend (0.0-1.0, higher = more enhancement)
    filter_type: 'bandpass', 'highpass', or 'weighted_bandpass' (for smoke emphasis)
    smoke_emphasis: if True, weight mid-frequencies more (where smoke patterns live)
    save_spectrum: if True, save frequency spectrum visualization
    spectrum_path: path to save spectrum image (only used if save_spectrum=True)
    """
    if keep_color:
        chans = cv2.split(img)
        out_chans = []
        # Use green channel for spectrum visualization (middle channel, good representation)
        for i, ch in enumerate(chans):
            spec_path = spectrum_path if (save_spectrum and i == 1) else None
            filtered = _enhance_channel_fft(ch, low_radius, high_radius, filter_type, smoke_emphasis, 
                                           save_spectrum=(save_spectrum and i == 1), spectrum_path=spec_path)
            # Blend filtered features back into original
            blended = cv2.addWeighted(ch, 1.0 - blend_factor, filtered, blend_factor, 0)
            out_chans.append(blended)
        enhanced = cv2.merge(out_chans)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = _enhance_channel_fft(gray, low_radius, high_radius, filter_type, smoke_emphasis, 
                                       save_spectrum=save_spectrum, spectrum_path=spectrum_path)
        # Blend filtered features back into original grayscale
        blended = cv2.addWeighted(gray, 1.0 - blend_factor, filtered, blend_factor, 0)
        enhanced = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
    return enhanced

def visualize_fft_spectrum(channel, mask, output_path):
    """Visualize and save the frequency domain spectrum."""
    # Compute FFT
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Apply log scale for better visualization (add 1 to avoid log(0))
    magnitude_log = np.log(magnitude + 1)
    
    # Normalize to 0-255
    magnitude_log = magnitude_log - magnitude_log.min()
    if magnitude_log.max() > 0:
        magnitude_log = magnitude_log / magnitude_log.max()
    magnitude_log = (magnitude_log * 255).astype(np.uint8)
    
    # Create visualization with 3 panels: original spectrum, mask, filtered spectrum
    h, w = magnitude_log.shape
    vis = np.zeros((h, w * 3), dtype=np.uint8)
    
    # Panel 1: Original spectrum
    vis[:, :w] = magnitude_log
    
    # Panel 2: Filter mask (normalized to 0-255)
    mask_vis = (mask * 255).astype(np.uint8)
    vis[:, w:2*w] = mask_vis
    
    # Panel 3: Filtered spectrum
    fshift_filtered = fshift * mask
    magnitude_filtered = np.abs(fshift_filtered)
    magnitude_filtered_log = np.log(magnitude_filtered + 1)
    magnitude_filtered_log = magnitude_filtered_log - magnitude_filtered_log.min()
    if magnitude_filtered_log.max() > 0:
        magnitude_filtered_log = magnitude_filtered_log / magnitude_filtered_log.max()
    magnitude_filtered_log = (magnitude_filtered_log * 255).astype(np.uint8)
    vis[:, 2*w:] = magnitude_filtered_log
    
    # Add labels (simple text overlay)
    cv2.putText(vis, "Original Spectrum", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(vis, "Filter Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(vis, "Filtered Spectrum", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    cv2.imwrite(output_path, vis)
    return vis

def _enhance_channel_fft(channel, low_radius, high_radius, filter_type='bandpass', smoke_emphasis=True, 
                         save_spectrum=False, spectrum_path=None):
    # channel: 2D single-channel uint8
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    
    # Choose filter type
    if filter_type == 'highpass':
        # High-pass: emphasizes edges/structures (smoke), reduces uniform areas (snow backgrounds)
        # Lower cutoff = more aggressive (removes more low-freq snow)
        mask = highpass_mask(channel.shape, cutoff_radius=low_radius, order=2)
    elif filter_type == 'weighted_bandpass':
        # Weighted bandpass: emphasizes mid-frequencies where smoke patterns live
        mask = weighted_bandpass_mask(channel.shape, low_radius, high_radius, smoke_emphasis)
    else:  # 'bandpass'
        mask = bandpass_mask(channel.shape, low_radius, high_radius)
    
    # Save spectrum visualization if requested
    if save_spectrum and spectrum_path:
        visualize_fft_spectrum(channel, mask, spectrum_path)
    
    fshift_filtered = fshift * mask
    # inverse
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # normalize to 0..255
    img_back = img_back - img_back.min()
    if img_back.max() > 0:
        img_back = img_back / img_back.max()
    img_back = (img_back * 255).astype(np.uint8)
    # optional: enhance contrast a bit (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_back = clahe.apply(img_back)
    return img_back

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_folder = os.path.join(script_dir, "dataset/faint_smoke")
    output_folder = os.path.join(script_dir, "dataset/faint_smoke_enhanced")
    spectrum_folder = os.path.join(script_dir, "dataset/faint_smoke_spectrum")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(spectrum_folder, exist_ok=True)

    # Filter parameters - tuned for smoke vs snow
    # For smoke enhancement: emphasize mid-frequencies, suppress high-freq snow sparkle
    low_radius = 5     # removes very low frequencies (large smooth areas, snow backgrounds)
    high_radius = 40   # removes very high frequencies (snow sparkle, fine noise)
    blend_factor = 0.4  # how much enhancement to blend (0.0 = original, 1.0 = fully filtered)
    
    # Filter type options:
    # 'bandpass' - standard bandpass filter
    # 'highpass' - emphasizes edges/structures (good for smoke), reduces uniform areas
    # 'weighted_bandpass' - emphasizes mid-frequencies where smoke patterns live
    filter_type = 'weighted_bandpass'  # Try: 'highpass', 'bandpass', or 'weighted_bandpass'
    smoke_emphasis = True  # Weight mid-frequencies more (where smoke lives)

    files = sorted(glob(os.path.join(input_folder, "*.jpg")))
    for fp in tqdm(files):
        img = cv2.imread(fp)
        base = os.path.basename(fp)
        base_name = os.path.splitext(base)[0]
        spectrum_path = os.path.join(spectrum_folder, f"{base_name}_spectrum.jpg")
        
        enhanced = enhance_image_bandpass(
            img, low_radius, high_radius, 
            keep_color=True, 
            blend_factor=blend_factor,
            filter_type=filter_type,
            smoke_emphasis=smoke_emphasis,
            save_spectrum=True,
            spectrum_path=spectrum_path
        )
        cv2.imwrite(os.path.join(output_folder, base), enhanced)

    print("Done. Enhanced images saved to:", output_folder)
    print("Frequency spectra saved to:", spectrum_folder)
