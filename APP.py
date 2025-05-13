import os
import tempfile
from PIL import Image
import streamlit as st
import ocr_processor

# Streamlit configuration
st.set_page_config(page_title='Egyptian ID Card OCR', page_icon='💳', layout='wide')

# Initialize session state for navigation
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Home"

# Sidebar navigation menu
# Removed "Guide" from the tabs list
tabs = ["Home"]
selected_tab = st.sidebar.selectbox("Navigation", tabs, index=tabs.index(st.session_state.current_tab),
                                    key="nav_selectbox")

# Update the session state with the selected tab
# This block will effectively not do much if there's only one tab, but it's harmless to keep
if st.session_state.current_tab != selected_tab:
    st.session_state.current_tab = selected_tab
    st.experimental_rerun()  # Use st.rerun() in newer Streamlit versions if available

# Home Tab
if st.session_state.current_tab == "Home":
    st.title("💳 Egyptian ID Card Information Extractor")
    st.markdown("Upload an image of an Egyptian ID card to extract information.")

    uploaded_file = st.file_uploader("Upload an ID card image",
                                     type=['webp', 'jpg', 'tif', 'tiff', 'png', 'mpo', 'bmp', 'jpeg', 'dng', 'pfm'],
                                     key="file_uploader")

    # If no file is uploaded, display the HOME image placeholder
    if not uploaded_file:
        try:
            st.image("ocr2.png", caption="Please upload an ID card image.", use_container_width=True)
        except FileNotFoundError:
            st.info("Welcome! Please upload an ID card image to begin. (Note: ocr2.png placeholder image not found)")
        except Exception as e:
            st.info(f"Welcome! Please upload an ID card image to begin. (Error loading placeholder ocr2.png: {e})")
    else:
        # If a file is uploaded, process it
        temp_file_path = None  # Initialize to ensure it's defined for finally block
        try:
            # Create a temporary file with the correct extension
            file_extension = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            pil_image = Image.open(temp_file_path)
            # Corrected: use_container_width instead of use_column_width
            st.sidebar.image(pil_image, caption="Uploaded ID Card", use_container_width=True)

            st.subheader('Processing Results:')
            with st.spinner("Processing ID card... This may take a moment."):
                # Call the detect_and_process_id_card function from ocr_processor
                extracted_data = ocr_processor.detect_and_process_id_card(temp_file_path)

            if extracted_data:
                try:
                    # Attempt to display d2.jpg after processing
                    st.image("d2.jpg", use_container_width=True)
                except FileNotFoundError:
                    st.markdown("---")
                    st.caption("(Note: d2.jpg image not found, so a separator is shown instead.)")
                except Exception as e:
                    st.markdown("---")
                    st.caption(f"(Note: Could not load d2.jpg: {e})")

                st.markdown(" ## Extracted Information:")
                if len(extracted_data) == 7:
                    national_id, birth_date, governorate, gender, conf, full_name, address = extracted_data

                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**National ID:** {national_id if national_id else 'Not detected'}")
                        st.info(f"**Full Name:** {full_name if full_name else 'Not detected'}")
                        st.info(f"**Address:** {address if address else 'Not detected'}")
                    with col2:
                        st.success(f"**Birth Date:** {birth_date if birth_date else 'Not detected'}")
                        st.success(f"**Governorate:** {governorate if governorate else 'Not detected'}")
                        st.success(f"**Gender:** {gender if gender else 'Not detected'}")
                    st.metric(label="NID Detection Confidence", value=f"{conf}%" if conf else "N/A")

                elif len(extracted_data) == 5:
                    job, expiry, status, issue, expiry_conf = extracted_data
                    st.markdown("#### Secondary Information (e.g., Back of Card):")
                    st.info(f"**Job:** {job if job else 'Not detected'}")
                    st.info(f"**Expiry Date:** {expiry if expiry else 'Not detected'}")
                    st.info(f"**Marital Status/Demo:** {status if status else 'Not detected'}")
                    st.info(f"**Issue Place/Date:** {issue if issue else 'Not detected'}")
                else:
                    st.error("Received unexpected data format from processing.")
            else:
                st.warning(
                    "Could not extract information. The ID card might not be clear, no card was detected, or an error occurred during processing.")

        except AttributeError as ae:
            if "'NoneType' object has no attribute 'Reader'" in str(ae) or (hasattr(ocr_processor, 'reader') and ocr_processor.reader is None):
                st.error(
                    "OCR engine (EasyOCR Reader) could not be initialized. Please check the console for errors when `ocr_processor.py` was imported or run. This usually happens if EasyOCR failed to load its models.")
            else:
                st.error(f"An AttributeError occurred: {ae}")
        except Exception as e:
            st.error(f"An unexpected error occurred in the Streamlit app: {e}")
            st.error(
                "Please ensure all model files (.pt) are in the correct directory and the image is valid. Check the console for more details.")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

