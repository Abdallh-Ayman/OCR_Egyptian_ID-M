from ultralytics import YOLO
import cv2
import easyocr
import matplotlib.pyplot as plt

# Initialize EasyOCR reader (do this once)
try:
    reader = easyocr.Reader(['ar'], gpu=False)
except Exception as e:
    print(f"Error initializing EasyOCR Reader: {e}")
    reader = None

def preprocess_image(cropped_image):
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    return gray_image

def extract_text(image, bbox, lang='ara'):
    if reader is None:
        print("EasyOCR Reader not initialized. Cannot extract text.")
        return ""

    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]

    # Ensure cropped image is not empty
    if cropped_image.size == 0:
        print(f"Warning: Cropped image for bbox {bbox} is empty.")
        return ""

    preprocessed_image = preprocess_image(cropped_image)

    # print(f"Extracting text from bbox: {bbox}")
    # show_image(cropped_image, title="Cropped Field (Original BGR)")
    # show_image(preprocessed_image, title="Cropped Field (Grayscale)")

    try:
        results = reader.readtext(preprocessed_image, detail=0, paragraph=True)
        return ' '.join(results).strip()
    except Exception as e:
        print(f"Error during EasyOCR text extraction for bbox {bbox}: {e}")
        return ""


def detect_national_id(cropped_image):
    # Ensure cropped image is not empty
    if cropped_image.size == 0:
        print("Warning: Cropped image for national ID detection is empty.")
        return ""

    model = YOLO('detect_id.pt')  # Make sure 'detect_id.pt' is in the same directory or provide full path
    results = model(cropped_image, verbose=False)  # Added verbose=False to reduce console output
    detected_info = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, _, _, _ = map(int, box.xyxy[0])
            detected_info.append((cls, x1))

    detected_info.sort(key=lambda x: x[1])
    return ''.join([str(cls) for cls, _ in detected_info])


def expand_bbox_custom(bbox, scale_w=1.0, scale_h=1.0, image_shape=None):
    """
    Expand a bounding box separately for width and height.

    Args:
        bbox: [x_min, y_min, x_max, y_max]
        scale_w: Width expansion factor
        scale_h: Height expansion factor
        image_shape: (height, width, channels)

    Returns:
        Expanded [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    center_x = x_min + width / 2
    center_y = y_min + height / 2

    new_width = width * scale_w
    new_height = height * scale_h

    new_x_min = int(center_x - new_width / 2)
    new_y_min = int(center_y - new_height / 2)
    new_x_max = int(center_x + new_width / 2)
    new_y_max = int(center_y + new_height / 2)

    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(img_w - 1, new_x_max)
        new_y_max = min(img_h - 1, new_y_max)

    # Ensure coordinates are valid
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = max(new_x_min, new_x_max)
    new_y_max = max(new_y_min, new_y_max)
    if image_shape:
        new_x_max = min(image_shape[1], new_x_max)
        new_y_max = min(image_shape[0], new_y_max)

    return [new_x_min, new_y_min, new_x_max, new_y_max]


def expand_bbox_more_custom(
        bbox,
        x_min_shift=0.0, x_max_shift=0.0,
        y_min_shift=0.0, y_max_shift=0.0,
        image_shape=None
):
    """
    Expand or contract a bounding box with independent control over each side.

    Args:
        bbox: [x_min, y_min, x_max, y_max]
        x_min_shift: Amount to subtract from x_min (positive = expand left)
        x_max_shift: Amount to add to x_max (positive = expand right)
        y_min_shift: Amount to subtract from y_min (positive = expand upward)
        y_max_shift: Amount to add to y_max (positive = expand downward)
        image_shape: (height, width, channels), optional, for clipping

    Returns:
        Adjusted bounding box as [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Prevent division by zero if width or height is 0
    width = width if width > 0 else 1
    height = height if height > 0 else 1

    new_x_min = x_min - int(x_min_shift * width)
    new_x_max = x_max + int(x_max_shift * width)
    new_y_min = y_min - int(y_min_shift * height)
    new_y_max = y_max + int(y_max_shift * height)

    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(img_w - 1, new_x_max)
        new_y_max = min(img_h - 1, new_y_max)

    # Ensure coordinates are valid
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = max(new_x_min, new_x_max)
    new_y_max = max(new_y_min, new_y_max)
    if image_shape:
        new_x_max = min(image_shape[1], new_x_max)
        new_y_max = min(image_shape[0], new_y_max)

    return [new_x_min, new_y_min, new_x_max, new_y_max]

def decode_egyptian_id(id_number):
    governorates = {
        '01': 'Cairo', '02': 'Alexandria', '03': 'Port Said', '04': 'Suez', '11': 'Damietta',
        '12': 'Dakahlia', '13': 'Ash Sharqia', '14': 'Kaliobeya', '15': 'Kafr El - Sheikh',
        '16': 'Gharbia', '17': 'Monoufia', '18': 'El Beheira', '19': 'Ismailia', '21': 'Giza',
        '22': 'Beni Suef', '23': 'Fayoum', '24': 'El Menia', '25': 'Assiut', '26': 'Sohag',
        '27': 'Qena', '28': 'Aswan', '29': 'Luxor', '31': 'Red Sea', '32': 'New Valley',
        '33': 'Matrouh', '34': 'North Sinai', '35': 'South Sinai', '88': 'Foreign'
    }

    if not isinstance(id_number, str) or len(id_number) < 13 or not id_number.isdigit():  # Added isinstance check
        return {
            'Birth Date': 'Invalid',
            'Governorate': 'Invalid',
            'Gender': 'Invalid'
        }

    try:
        century_digit = int(id_number[0])
        year = int(id_number[1:3])
        month = int(id_number[3:5])
        day = int(id_number[5:7])
        governorate_code = id_number[7:9]
        gender_code = int(id_number[12])  # This should be id_number[12], not id_number[-2] for 14-digit ID
    except (ValueError, IndexError) as e:
        print(f"Error decoding ID '{id_number}': {e}")
        return {
            'Birth Date': 'Invalid',
            'Governorate': 'Invalid',
            'Gender': 'Invalid'
        }

    # Basic validation for date components
    if not (1 <= month <= 12 and 1 <= day <= 31):
        birth_date_str = "Invalid Date"
    else:
        full_year = 1900 + year if century_digit == 2 else 2000 + year if century_digit == 3 else 0
        if full_year == 0:  # Invalid century digit
            birth_date_str = "Invalid Date (Century)"
        else:
            birth_date_str = f"{full_year:04d}-{month:02d}-{day:02d}"
            # Further date validation could be added here (e.g., check if day is valid for month/year)

    gender = "Male" if gender_code % 2 != 0 else "Female"
    governorate = governorates.get(governorate_code, "Unknown")

    return {
        'Birth Date': birth_date_str,
        'Governorate': governorate,
        'Gender': gender
    }


def show_image(image, title="Image"):
    # This function will open a new window if called.
    # In a web app context, this might not be ideal.
    # Consider commenting out calls to this if pop-ups are an issue.
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error in show_image: {e}")


def process_image(cropped_image):
    # Ensure cropped image is not empty
    if cropped_image.size == 0:
        print("Warning: Main cropped image for processing is empty.")
        return None  # Or return an appropriate empty/error structure

    model = YOLO('detect_odjects.pt')  # Make sure 'detect_odjects.pt' is in the same directory
    results = model(cropped_image, verbose=False)  # Added verbose=False

    nid, conf = "", 0.0  # conf should be float
    full_name = address = job = expiry = status = issue = ""
    expiry_conf = 0.0  # This was defined but not used later, keeping for consistency

    detected_fields_image = cropped_image.copy()  # Work on a copy for drawing


    for result_item in results:  # Renamed result to result_item to avoid conflict with module
        for box in result_item.boxes:
            bbox = [int(coord) for coord in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            class_name = result_item.names[class_id]
            confidence = float(box.conf[0].item())
            label = ""  # f"{class_name} ({confidence:.2f})" # Label not shown in original code

            # Default: no expansion
            expanded_bbox = bbox

            if class_name in( 'nid' , 'invalid_nid'):
                # Ensure bbox is valid before expansion
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    expanded_bbox = expand_bbox_more_custom(bbox, x_min_shift=0.1, x_max_shift=0.1, y_min_shift=.3,
                                                            y_max_shift=.01, image_shape=cropped_image.shape)
                    field_to_process = cropped_image[expanded_bbox[1]:expanded_bbox[3],
                                       expanded_bbox[0]:expanded_bbox[2]]
                    nid = detect_national_id(field_to_process)
                    conf = round(confidence * 100.0, 2)
                else:
                    print(f"Skipping NID due to invalid initial bbox: {bbox}")

            elif class_name in(  'lastName' , 'invalid_lastName') :  # This seems to be used for full_name
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    expanded_bbox = expand_bbox_more_custom(bbox, x_min_shift=0.09, x_max_shift=0.09, y_min_shift=1.2,
                                                            y_max_shift=0.04, image_shape=cropped_image.shape)
                    full_name = extract_text(cropped_image, expanded_bbox, lang='ara')
                else:
                    print(f"Skipping lastName due to invalid initial bbox: {bbox}")

            elif class_name in(  'address' ,'invalid_address'):
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    expanded_bbox = expand_bbox_custom(bbox, scale_w=1.2, scale_h=1.1, image_shape=cropped_image.shape)
                    address = extract_text(cropped_image, expanded_bbox, lang='ara')
                else:
                    print(f"Skipping address due to invalid initial bbox: {bbox}")

            elif class_name == 'job':
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    expanded_bbox = expand_bbox_more_custom(bbox, x_min_shift=1.2, x_max_shift=0.1, y_min_shift=0.15,
                                                            y_max_shift=1.3, image_shape=cropped_image.shape)
                    job = extract_text(cropped_image, expanded_bbox, lang='ara')
                else:
                    print(f"Skipping job due to invalid initial bbox: {bbox}")

            elif class_name == 'expiry':
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    expanded_bbox = expand_bbox_more_custom(bbox, x_min_shift=0.3, x_max_shift=0.1, y_min_shift=0.3,
                                                            y_max_shift=0.3, image_shape=cropped_image.shape)
                    expiry = extract_text(cropped_image, expanded_bbox, lang='ara')
                else:
                    print(f"Skipping expiry due to invalid initial bbox: {bbox}")

            elif class_name == 'demo':  # Assuming this is marital status or similar
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    expanded_bbox = expand_bbox_more_custom(bbox, x_min_shift=0.1, x_max_shift=0.1, y_min_shift=0.09,
                                                            y_max_shift=0.1, image_shape=cropped_image.shape)
                    status = extract_text(cropped_image, expanded_bbox, lang='ara')
                else:
                    print(f"Skipping demo due to invalid initial bbox: {bbox}")

            elif class_name == 'issue':  # Issue place or date
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    expanded_bbox = expand_bbox_more_custom(bbox, x_min_shift=0.1, x_max_shift=0.1, y_min_shift=0.1,
                                                            y_max_shift=0.1, image_shape=cropped_image.shape)
                    issue = extract_text(cropped_image, expanded_bbox, lang='ara')
                else:
                    print(f"Skipping issue due to invalid initial bbox: {bbox}")

            # Draw rectangle on the copied image if the box is valid
            if class_name != 'firstName':  # firstName is not processed for text extraction in the original logic
                if expanded_bbox[2] > expanded_bbox[0] and expanded_bbox[3] > expanded_bbox[1]:
                    cv2.rectangle(detected_fields_image, (expanded_bbox[0], expanded_bbox[1]),
                                  (expanded_bbox[2], expanded_bbox[3]), (0, 255, 0), 2)
                    # cv2.putText(detected_fields_image, label, (expanded_bbox[0], expanded_bbox[1] - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Label display was commented out

    # show_image(detected_fields_image, title="Detected Fields on Card") # Call show_image with the image that has rectangles

    if nid:  # If National ID is detected, assume it's the front of the card
        decoded_info = decode_egyptian_id(nid)
        return (
            nid,
            decoded_info["Birth Date"],
            decoded_info["Governorate"],
            decoded_info["Gender"],
            conf,  # NID detection confidence
            full_name,  # Extracted full name
            address  # Extracted address
        )
    else:  # Otherwise, assume it might be the back or other information is primary
        return (job, expiry, status, issue, expiry_conf)  # expiry_conf is always 0.0 as per current logic




def detect_and_process_id_card(image_path, id_card_confidence_threshold=0.7):
    INVALID_ID_MESSAGE = "Input does not appear to be a valid Egyptian ID card."

    """
    Detects an Egyptian ID card, processes it, and validates the result.

    Args:
        image_path (str): Path to the input image.
        id_card_confidence_threshold (float): Minimum confidence for the initial ID card detection.

    Returns:
        tuple: Extracted data if successful (7 elements for front, 5 for back).
        str: INVALID_ID_MESSAGE if no valid Egyptian ID is detected or processed.
        None: If there was an image reading error.
    """
    # Check if EasyOCR initialized correctly
    if reader is None:
        print("EasyOCR Reader not initialized. Cannot process.")
        # You might want to return a specific error message or raise an exception here
        # For simplicity, we'll let it potentially fail later or return the generic invalid message.
        # Consider returning "ERROR: OCR engine not ready."

    try:
        id_card_model = YOLO('detect_id_card.pt') # Load model inside try block if path might be wrong
    except Exception as e:
        print(f"Error loading ID card detection model (detect_id_card.pt): {e}")
        return "ERROR: ID Card detection model failed to load." # More specific error

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from path: {image_path}")
            return None # Indicate image read failure specifically
    except Exception as e:
        print(f"Exception reading image {image_path}: {e}")
        return None # Indicate image read failure specifically

    #show_image(image, title="Cropped Field (Original BGR)")
    id_card_results = id_card_model(image, verbose=False)
    #show_image(id_card_results, title="Cropped Field (Original BGR)")

    best_detection = None # Store the best candidate detection

    for result_item in id_card_results:
        for box in result_item.boxes:
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            class_name = result_item.names[class_id] # Get class name

            # **Crucial Check 1: Class Name (If applicable) and Confidence**
            # *Modify this 'if' condition if your model has specific class names*
            # Example: if class_name in ['egyptian_id_front', 'egyptian_id_back'] and confidence >= id_card_confidence_threshold:
            # If your model only has a generic 'id_card' class, just use confidence:
            if confidence >= id_card_confidence_threshold:
                 # Optional: Check if class_name suggests it's a card type if model supports it
                 print(f"Detected potential card '{class_name}' with confidence {confidence:.2f}")
                 if best_detection is None or confidence > best_detection['confidence']:
                    best_detection = {'box': box, 'confidence': confidence, 'image_shape': image.shape}
            else:
                 print(f"Skipping detection with low confidence {confidence:.2f}")


    if best_detection:
        box = best_detection['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Ensure coordinates are valid before cropping
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            print(f"Invalid bounding box for detected ID card: {[x1, y1, x2, y2]}. Skipping.")
            return INVALID_ID_MESSAGE # Treat as invalid if box is bad

        cropped_image = image[y1:y2, x1:x2]


        if cropped_image.size == 0:
            print(f"Warning: Cropped ID card image is empty for box {[x1, y1, x2, y2]}.")
            return INVALID_ID_MESSAGE # Treat as invalid if crop is empty

        # Process the cropped potential ID card
        processed_data = process_image(cropped_image)

        # **Crucial Check 2: Validate the processed data structure and content**
        if processed_data:
            is_valid = False
            if len(processed_data) == 7: # Expected front card structure
                nid, birth_date, gov, gender, conf, name, addr = processed_data
                # Check if key fields were actually extracted (not empty or just noise)
                if nid and len(nid) >= 13 and name: # Require NID and Name at minimum for front
                     print("Validation: Detected Front ID structure with key fields.")
                     is_valid = True
                else:
                    INVALID_ID_MESSAGE ="Please give me another clear image as Validation Failed: Front structure missing key fields (NID/Name)."
                    return INVALID_ID_MESSAGE
            elif len(processed_data) == 5: # Expected back card structure
                job, expiry, status, issue, expiry_conf = processed_data
                # Check if key fields were actually extracted
                if job or expiry: # Require Job or Expiry at minimum for back
                    print("Validation: Detected Back ID structure with key fields.")
                    is_valid = True
                else:
                    print("Validation Failed: Back structure missing key fields (Job/Expiry).")
            else:
                 print("Validation Failed: Unexpected data structure returned by process_image.")


            if is_valid:
                return processed_data # Return the valid tuple
            else:
                return INVALID_ID_MESSAGE # Return rejection message if validation fails

        else:
            # process_image returned None or empty
            print("Processing the cropped image failed to return data.")
            return INVALID_ID_MESSAGE

    else:
        # No detection met the confidence/class criteria
        print(f"No suitable ID card detected meeting the threshold ({id_card_confidence_threshold}).")
        return INVALID_ID_MESSAGE


# --- [Keep your main function, but update its result handling] ---

def main():
    # Example usage:
    # Use one of the provided example images
    image_file = r"E:\ITI BI\we_tasks\all_files\test1.jpg" # Replace with actual path
    # Or test with a non-ID image:
    # image_file = "path/to/your/non_id_image.jpg"

    print(f"Processing image: {image_file}")
    result_data = detect_and_process_id_card(image_file, id_card_confidence_threshold=.6) # Adjust threshold as needed

    # Updated result handling
    if isinstance(result_data, tuple): # Check if we got a valid data tuple
        if len(result_data) == 7:
            id_number, birth_date, governorate, gender, conf_score, full_name_val, address_val = result_data
            print("\n--- Extracted ID Card Front Information ---")
            print(f"National ID: {id_number}")
            print(f"Confidence Score (NID detection): {conf_score}%")
            print(f"Full Name: {full_name_val}")
            print(f"Address: {address_val}")
            print(f"Birth Date: {birth_date}")
            print(f"Governorate: {governorate}")
            print(f"Gender: {gender}")
        elif len(result_data) == 5:
            job_val, expiry_val, status_val, issue_val, expiry_conf_score = result_data
            print("\n--- Extracted ID Card Back Information ---")
            print(f"Job: {job_val}")
            print(f"Expiry Date: {expiry_val}")
            print(f"Marital Status/Demo: {status_val}")
            print(f"Issue Place/Date: {issue_val}")
            # print(f"Expiry Confidence: {expiry_conf_score}")
        # No need for an else here because we checked isinstance(tuple)

    elif isinstance(result_data, str): # Check if it's the error/rejection message
        print(f"\nResult: {result_data}") # Print the message (e.g., INVALID_ID_MESSAGE)
    elif result_data is None:
        print("\nError: Could not read the input image file.")
    else:
        print("\n--- Unexpected result format ---")
        print(result_data)


if __name__ == "__main__":
    # Ensure EasyOCR is ready before running main
    if reader is None:
         print("Cannot run main: EasyOCR Reader failed to initialize.")
    else:
         main()
