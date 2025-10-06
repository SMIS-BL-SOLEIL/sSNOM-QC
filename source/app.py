import os
import datetime
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from contextlib import contextmanager

from pySNOM import readers
from matplotlib.ticker import FormatStrFormatter


# Configuration constants
START_WN1, END_WN1 = 800, 1300
START_WN2, END_WN2 = 650, 1800
DEMOD_OPTIONS = ["O2A", "O3A", "O4A", "O5A"]
MAX_FILES = 2

# Set page config first
st.set_page_config(layout="wide", page_title="sSNOM-QC")


@contextmanager
def temp_file_context(uploaded_file):
    """Context manager for temporary file handling."""
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=os.path.splitext(uploaded_file.name)[1]
    )
    try:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        yield tmp_file.name
    finally:
        os.unlink(tmp_file.name)


@st.cache_data(show_spinner=False)
def load_nea(file_name, file_bytes):
    """Load NeaSNOM data with caching."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        nea_reader = readers.NeaSpectralReader(tmp_file_path)
        nea_data, nea_measparams = nea_reader.read()
        return nea_data, nea_measparams
    finally:
        os.unlink(tmp_file_path)


def init_session_state():
    """Initialize session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'upload_widget_key' not in st.session_state:
        st.session_state.upload_widget_key = 0
    if 'show_motd' not in st.session_state:
        st.session_state.show_motd = True



def reset_app():
    """Reset application state."""
    st.session_state.uploaded_files = []
    st.session_state.upload_widget_key += 1
    st.session_state.show_motd = True



def handle_file_upload(uploaded_file):
    """Handle file upload with validation."""
    if len(st.session_state.uploaded_files) >= MAX_FILES:
        st.warning(f"Maximum of {MAX_FILES} files allowed. Use reset button to clear.")
        return False
    
    # Check if file already uploaded
    if any(f['name'] == uploaded_file.name for f in st.session_state.uploaded_files):
        return False
    
    try:
        with st.spinner(f"Loading {uploaded_file.name}..."):
            data, measparams = load_nea(uploaded_file.name, uploaded_file.getvalue())
            
            st.session_state.uploaded_files.append({
                'name': uploaded_file.name,
                'data': data,
                'measparams': measparams
            })
            
            # Increment key to clear the uploader
            st.session_state.upload_widget_key += 1
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return False


def calculate_snr_stats(wn, ratio, start_wn, end_wn):
    """Calculate SNR and plot limits for a given range."""
    mask = (wn >= start_wn) & (wn <= end_wn)
    mean_ratio = np.mean(ratio[mask])
    std_ratio = np.std(ratio[mask])
    
    return {
        'snr': mean_ratio / std_ratio,
        'y_min': np.min(ratio[mask]) - std_ratio,
        'y_max': np.max(ratio[mask]) + std_ratio,
        'mask': mask
    }


# @st.cache_resource
def setup_plot_style():
    """Setup matplotlib style (cached to avoid reloading)."""
    try:
        plt.style.use("source/plot-style.mplstyle")
    except:
        pass  # Use default style if custom not found


def create_comparison_plot(file_data_1, file_data_2, order):
    """Create the comparison plot figure."""
    sp1 = file_data_1['data'][order]
    wn1 = file_data_1['data']["Wavenumber"]
    sp2 = file_data_2['data'][order]
    wn2 = file_data_2['data']["Wavenumber"]
    
    # Validate spectra length
    if len(sp1) != len(sp2):
        raise ValueError("Spectra must have the same length.")
    
    # Calculate ratio
    ratio = sp1 / sp2
    
    # Calculate statistics for both ranges
    stats1 = calculate_snr_stats(wn1, ratio, START_WN1, END_WN1)
    stats2 = calculate_snr_stats(wn1, ratio, START_WN2, END_WN2)
    
    # Create figure
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.7, 0.3], height_ratios=[1, 1])
    
    # Left subplot: full spectra
    ax_left = plt.subplot(gs[:, 0])
    ax_left.plot(wn1, sp1, label=f"{file_data_1['name']} {order}")
    ax_left.plot(wn2, sp2, label=f"{file_data_2['name']} {order}")
    ax_left.set_xlim(0, 5000)
    ax_left.set_xlabel("Frequency / cm⁻¹")
    ax_left.set_ylabel(f"{order} / a.u.")
    ax_left.legend(loc="upper right")
    
    now_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
    ax_left.set_title(
        f"Project: {file_data_1['measparams']['Project']}\nPlot date: {now_str}",
        loc="left"
    )
    
    # Top-right subplot: first range
    ax_right_top = plt.subplot(gs[0, 1])
    ax_right_top.plot(wn1, ratio, label=f"SNR: {stats1['snr']:.1f}", color="#28ad2c")
    ax_right_top.set_xlim(START_WN1, END_WN1)
    ax_right_top.set_ylim(stats1['y_min'], stats1['y_max'])
    ax_right_top.set_ylabel(f"{order} Ratio / a.u.")
    ax_right_top.legend(loc="upper right")
    ax_right_top.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Bottom-right subplot: second range
    ax_right_bottom = plt.subplot(gs[1, 1])
    ax_right_bottom.plot(wn1, ratio, label=f"SNR: {stats2['snr']:.1f}", color="#e0147a")
    ax_right_bottom.set_xlim(START_WN2, END_WN2)
    ax_right_bottom.set_ylim(stats2['y_min'], stats2['y_max'])
    ax_right_bottom.set_xlabel("Frequency / cm⁻¹")
    ax_right_bottom.set_ylabel(f"{order} Ratio / a.u.")
    ax_right_bottom.legend(loc="upper right")
    ax_right_bottom.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Add figure caption
    add_figure_caption(fig, file_data_1, file_data_2)
    
    st.session_state.show_motd = False

    return fig


def add_figure_caption(fig, file_data_1, file_data_2):
    """Add caption with measurement parameters."""
    def format_params(params):
        return (
            f"Exp. Date: {params['Date']}" 
            f"TA: {params['TipAmplitude']} nm - Avg: {params['Averaging']} - "
            f"Int time: {params['Integrationtime']} ms - "
            f"Interferometer: {params['InterferometerCenterDistance'][0]}, "
            f"{params['InterferometerCenterDistance'][1]}"
        )
    
    caption = (
        f"{file_data_1['name']}\n  {format_params(file_data_1['measparams'])}\n\n"
        f"{file_data_2['name']}\n  {format_params(file_data_2['measparams'])}"
    )
    
    fig.text(0.1, -0.15, caption, ha='left', va='bottom', fontsize=12)


def render_sidebar():
    """Render sidebar UI."""
    st.title("sSNOM-QC")
    st.write("Upload two NeaSNOM files.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "uploader",
        type=["txt"],
        key=f"uploader_{st.session_state.upload_widget_key}",
        label_visibility="collapsed"
    )
    
    # Handle upload
    if uploaded_file is not None:
        if handle_file_upload(uploaded_file):
            st.rerun()
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.markdown("### Uploaded Files:")
        for i, file_info in enumerate(st.session_state.uploaded_files, 1):
            st.write(f"**{i}.** {file_info['name']}")
    
    # Status and controls
    num_files = len(st.session_state.uploaded_files)
    
    if num_files == MAX_FILES:
        st.success("Both files uploaded.")
        order = st.segmented_control(
            "Select demodulation order",
            DEMOD_OPTIONS,
            selection_mode="single",
            width="content",
            default=DEMOD_OPTIONS[0]
        )
    elif num_files == 1:
        st.info("Upload one more file to proceed.")
        order = None
    else:
        st.info("Upload your first spectrum file.")
        order = None
    
    # Reset button
    if st.button("Reset All", type="primary", use_container_width=True):
        reset_app()
        st.rerun()
    
    return order


def render_metadata():
    """Render metadata section."""
    st.divider()
    st.write("### Full metadata")
    
    col1, col2 = st.columns(2)
    
    for col, file_data in zip([col1, col2], st.session_state.uploaded_files):
        with col:
            params = file_data['measparams']
            html = f"**{file_data['name']}**<br>"
            html += "<br>".join(f"<b>{k}:</b> {v}" for k, v in params.items())
            st.markdown(html, unsafe_allow_html=True)


def main():
    """Main application logic."""
    init_session_state()
    setup_plot_style()
    
    # Render sidebar and get selected order
    with st.sidebar:
        order = render_sidebar()
    
    motd = '''## Suggested experiment parameters

    For best comparison between measurements we suggest to use the following parameters.

    Tapping amplitude: 70 nm
    Number of acquisitions: 16
    Integration time: 20 ms
    Spectral resolution: 6 cm⁻¹
    '''
          # Hide motd once plot is successfully rendered

    motd_box = None
    if st.session_state.show_motd:
        motd_box = st.markdown(motd)

    # Main content
    if len(st.session_state.uploaded_files) == MAX_FILES and order:
        try:
            fig = create_comparison_plot(
                st.session_state.uploaded_files[0],
                st.session_state.uploaded_files[1],
                order
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)  # Free memory
            
            st.session_state.show_motd = False
            if motd_box:
                motd_box.empty()

            render_metadata()
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()