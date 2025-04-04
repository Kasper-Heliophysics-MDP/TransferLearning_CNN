import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_spectrogram(data, time_array=None, freq_array=None, time_range=None, freq_range=None, 
                     cmap='viridis', figsize=(10, 6), vmin=None, vmax=None):
    """
    Display a spectrogram with control over time and frequency ranges
    
    Parameters:
        data: Spectrogram data
        time_array: Time axis data
        freq_array: Frequency axis data
        time_range: Time range to display, format (start_time, end_time)
        freq_range: Frequency range to display, format (min_freq, max_freq)
        cmap: Color map
        figsize: Figure size
        vmin: Minimum data value for colormap scaling
        vmax: Maximum data value for colormap scaling
    """
    plt.figure(figsize=figsize)
    
    # Basic display with fixed color range
    im = plt.imshow(data.T, aspect='auto', cmap=cmap, origin='upper',
                   vmin=vmin, vmax=vmax)
    
    # If time array is provided, set more meaningful x-axis ticks
    if time_array is not None:
        # Calculate appropriate tick intervals
        total_points = len(time_array)
        tick_count = min(10, total_points)  # No more than 10 ticks
        step = total_points // tick_count
        positions = np.arange(0, total_points, step)
        
        if isinstance(time_array[0], datetime) or isinstance(time_array[0], str):
            # If time format, convert to more friendly labels
            time_labels = [time_array[i] if isinstance(time_array[i], str) 
                           else time_array[i].strftime('%H:%M:%S') for i in positions]
        else:
            # If numerical values, use directly
            time_labels = [str(time_array[i]) for i in positions]
            
        plt.xticks(positions, time_labels, rotation=45)
    
    # Control the displayed time range
    if time_range is not None:
        start_idx, end_idx = 0, len(data)
        
        # Find indices corresponding to time_range
        if time_array is not None:
            if isinstance(time_range[0], str) or isinstance(time_range[0], datetime):
                # If time_range is in time format, find the closest indices
                start_time = time_range[0]
                end_time = time_range[1]
                
                # Find the closest indices - using helper function
                # Assumes find_closest_time_index is available from SpectrogramSlicer class
                # Otherwise implement a similar search function
                if 'find_closest_time_index' in globals():
                    start_idx = find_closest_time_index(time_array, start_time)
                    end_idx = find_closest_time_index(time_array, end_time)
                else:
                    # Simple linear search
                    if isinstance(time_array[0], datetime):
                        start_idx = np.argmin([abs((t - start_time).total_seconds()) for t in time_array])
                        end_idx = np.argmin([abs((t - end_time).total_seconds()) for t in time_array])
                    else:
                        # Assume string format time
                        start_dt = datetime.strptime(start_time, "%H:%M:%S") if isinstance(start_time, str) else start_time
                        end_dt = datetime.strptime(end_time, "%H:%M:%S") if isinstance(end_time, str) else end_time
                        
                        time_array_dt = [datetime.strptime(t, "%H:%M:%S") if isinstance(t, str) else t for t in time_array]
                        start_idx = np.argmin([abs((t - start_dt).total_seconds()) for t in time_array_dt])
                        end_idx = np.argmin([abs((t - end_dt).total_seconds()) for t in time_array_dt])
            else:
                # Numerical indices
                start_idx = max(0, time_range[0])
                end_idx = min(len(data), time_range[1])
        
        plt.xlim(start_idx, end_idx)
    
    # Control the displayed frequency range
    if freq_range is not None:
        plt.ylim(freq_range[1], freq_range[0])
    else:
        plt.gca().invert_yaxis()
    
    plt.colorbar(im, label='Intensity')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def find_closest_time_index(time_array, target_time):
    """
    Find the index of the closest time value in time_array to the target_time
    
    Parameters:
        time_array: Array of time values (can be datetime objects or strings)
        target_time: Target time to find (can be datetime object or string)
        
    Returns:
        Index of the closest time in time_array
    """
    # Convert target_time to datetime if it's a string
    if isinstance(target_time, str):
        # Try multiple time formats for target_time
        formats = ["%H:%M:%S", "%H:%M:%S.%f"]
        for fmt in formats:
            try:
                target_dt = datetime.strptime(target_time, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unsupported time format for target_time: {target_time}")
    else:
        target_dt = target_time
    
    # Process based on time_array type
    if isinstance(time_array[0], datetime):
        # If time_array contains datetime objects
        return np.argmin([abs((t - target_dt).total_seconds()) for t in time_array])
    elif isinstance(time_array[0], str):
        # If time_array contains string time representations
        time_array_dt = []
        for t in time_array:
            # Try multiple time formats
            for fmt in ["%H:%M:%S", "%H:%M:%S.%f"]:
                try:
                    time_dt = datetime.strptime(t, fmt)
                    time_array_dt.append(time_dt)
                    break
                except ValueError:
                    continue
            else:
                # If no format works, use a default datetime value
                # or you could raise an error
                print(f"Warning: Could not parse time string: {t}")
                time_array_dt.append(datetime.min)
        
        return np.argmin([abs((t - target_dt).total_seconds()) for t in time_array_dt])
    else:
        # If time_array contains numeric values, just find closest match
        return np.argmin([abs(t - target_dt) for t in time_array])

# Usage:
# plot_spectrogram(data, time_array, freq_array=None, time_range=('18:45:00', '18:50:00'))