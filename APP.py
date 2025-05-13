# APP.py
import os
import tempfile
from PIL import Image
import streamlit as st
import ocr_processor  # Import your processing functions

# --- Streamlit Page Configuration ---
st.set_page_config(page_title='Egyptian ID Card OCR', page_icon='💳', layout='wide')

# --- Session State for Navigation (Optional but good practice) ---
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Home"

# --- Sidebar Navigation ---
# If you add more tabs later, add them to this list
tabs = ["Home"]
selected_tab = st.sidebar.selectbox("Navigation", tabs, index=tabs.index(st.session_state.current_tab),
                                    key="nav_selectbox")

# Update session state if tab changes (won't do much with only one tab)
if st.session_state.current_tab != selected_tab:
    st.session_state.current_tab = selected_tab
    # Use st.rerun() in newer Streamlit versions if available/needed
    st.experimental_rerun()

# --- Main Page Content ---
if st.session_state.current_tab == "Home":
    st.title("💳 Egyptian ID Card Information Extractor")
    st.markdown("""
    Upload an image of the **front or back** of an Egyptian National ID card.
    The system will attempt to detect the card, extract the information, and validate if it appears to be a genuine Egyptian ID.
    """)

    uploaded_file = st.file_uploader("Upload an ID card image",
                                     type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff'], # Common image types
                                     key="file_uploader")

    # --- Display Logic ---
    if not uploaded_file:
        # Display placeholder if no file is uploaded
        try:
            # Ensure ocr2.png is in your repository
            st.image("ocr2.png", caption="Please upload an ID card image.", use_container_width=True)
        except FileNotFoundError:
            st.info("Welcome! Please upload an ID card image to begin. (Note: ocr2.png placeholder image not found)")
        except Exception as e:
            st.info(f"Welcome! Please upload an ID card image to begin. (Error loading placeholder ocr2.png: {e})")
    else:
        # Process the uploaded file
        temp_file_path = None
        try:
            # Create a temporary file to save the upload
            file_extension = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Display uploaded image in sidebar
            try:
                pil_image = Image.open(temp_file_path)
                st.sidebar.image(pil_image, caption="Uploaded ID Card", use_container_width=True)
            except Exception as img_err:
                st.sidebar.error(f"Could not display uploaded image: {img_err}")

            st.subheader('Processing Results:')
            with st.spinner("Analyzing ID card... This might take a moment."):
                # --- Call the processing function ---
                # Adjust the confidence threshold as needed (0.0 to 1.0)
                extracted_data = ocr_processor.detect_and_process_id_card(temp_file_path, id_card_confidence_threshold=0.6)

            # --- Handle the results from the processor ---
            if isinstance(extracted_data, tuple):
                # SUCCESS: Data was extracted and validated as likely Egyptian ID
                try:
                    # Optional: Display secondary image if extraction successful
                    # Ensure d2.jpg is in your repository
                    st.image("d2.jpg", use_container_width=True)
                except FileNotFoundError:
                    st.markdown("---") # Separator if image not found
                    st.caption("(Note: d2.jpg display image not found)")
                except Exception as e:
                    st.markdown("---")
                    st.caption(f"(Note: Could not load d2.jpg: {e})")

                st.markdown(" ## Extracted Information:")

                if len(extracted_data) == 7:
                    # FRONT CARD DATA
                    national_id, birth_date, governorate, gender, conf, full_name, address = extracted_data
                    st.success("Detected **Front** of Egyptian ID Card.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**National ID:** {national_id if national_id else 'Not detected'}")
                        st.info(f"**Full Name:** {full_name if full_name else 'Not detected'}")
                        st.info(f"**Address:** {address if address else 'Not detected'}")
                    with col2:
                        st.success(f"**Birth Date:** {birth_date if birth_date else 'Not detected'}")
                        st.success(f"**Governorate:** {governorate if governorate else 'Not detected'}")
                        st.success(f"**Gender:** {gender if gender else 'Not detected'}")
                    # Display confidence of the NID *field* detection, not the initial card detection
                    st.metric(label="National ID Field Confidence", value=f"{conf:.1f}%" if conf else "N/A")

                elif len(extracted_data) == 5:
                    # BACK CARD DATA
                    job, expiry, status, issue, expiry_conf = extracted_data
                    st.success("Detected **Back** of Egyptian ID Card.")
                    st.markdown("#### Secondary Information:")
                    st.info(f"**Job:** {job if job else 'Not detected'}")
                    st.info(f"**Expiry Date:** {expiry if expiry else 'Not detected'}")
                    st.info(f"**Marital Status/Religion/Demo:** {status if status else 'Not detected'}") # Clarified potential content
                    st.info(f"**Issue Place/Date:** {issue if issue else 'Not detected'}")
                    # expiry_conf is likely 0.0 based on processor code, maybe hide or remove
                    # st.metric(label="Expiry Field Confidence", value=f"{expiry_conf:.1f}%" if expiry_conf else "N/A")

                else:
                    # This case should ideally be handled by the validation in ocr_processor
                    st.error("Processing completed, but data format is unexpected.")
                    st.json(extracted_data) # Show the raw data for debugging

            elif isinstance(extracted_data, str):
                # REJECTION or SPECIFIC ERROR from processor
                # This includes the INVALID_ID_MESSAGE or model loading errors
                st.warning(f"⚠️ {extracted_data}")
                # Optionally display the uploaded image even on rejection
                # try:
                #     pil_image = Image.open(temp_file_path)
                #     st.image(pil_image, caption="Uploaded Image (Rejected)", use_container_width=True)
                # except Exception:
                #     pass # Ignore if display fails

            elif extracted_data is None:
                # IMAGE READING FAILED
                st.error("❌ Error: Could not read or open the uploaded image file. Please try a different file.")

            else:
                # UNEXPECTED RETURN TYPE (Should not happen with current processor logic)
                st.error("❌ An unexpected internal error occurred during processing.")

        except AttributeError as ae:
             # Specific check for EasyOCR initialization failure
            if "'NoneType' object has no attribute 'Reader'" in str(ae) or \
               (hasattr(ocr_processor, 'reader') and ocr_processor.reader is None):
                st.error("❌ Critical Error: The OCR engine (EasyOCR) failed to initialize.")
                st.error("This might be due to deployment issues or missing model files for EasyOCR.")
                st.error("Please check the deployment logs ('Manage app').")
            else:
                st.error(f"❌ An unexpected attribute error occurred: {ae}")
                st.error("Please check the logs for more details.")

        except Exception as e:
            # General error catching
            st.error(f"❌ An unexpected error occurred: {e}")
            st.error("Please ensure the uploaded image is valid. Check logs if the problem persists.")
            # Consider adding more specific exception handling if needed

        finally:
            # --- Clean up the temporary file ---
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    # print(f"Removed temp file: {temp_file_path}") # For debugging
                except Exception as e:
                    print(f"Error removing temporary file {temp_file_path}: {e}") # Log error
