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

            if class_name == 'nid':
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

            elif class_name == 'lastName':  # This seems to be used for full_name
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    expanded_bbox = expand_bbox_more_custom(bbox, x_min_shift=0.09, x_max_shift=0.09, y_min_shift=1.2,
                                                            y_max_shift=0.04, image_shape=cropped_image.shape)
                    full_name = extract_text(cropped_image, expanded_bbox, lang='ara')
                else:
                    print(f"Skipping lastName due to invalid initial bbox: {bbox}")

            elif class_name == 'address':
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


def detect_and_process_id_card(image_path):
    id_card_model = YOLO('detect_id_card.pt')  # Make sure 'detect_id_card.pt' is in the same directory

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from path: {image_path}")
            return None
    except Exception as e:
        print(f"Exception reading image {image_path}: {e}")
        return None

    id_card_results = id_card_model(image, verbose=False)  # Pass image object, verbose=False

    # Process the first detected ID card if any
    for result_item in id_card_results:  # Renamed result to result_item
        if len(result_item.boxes) > 0:
            # Assuming the first detected box is the ID card
            box = result_item.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Ensure coordinates are valid before cropping
            if x1 >= x2 or y1 >= y2:
                print(f"Invalid bounding box for ID card: {[x1, y1, x2, y2]}. Skipping.")
                continue  # Or handle error appropriately

            cropped_image = image[y1:y2, x1:x2]

            if cropped_image.size == 0:
                print(f"Warning: Cropped ID card image is empty for box {[x1, y1, x2, y2]}.")
                continue

            return process_image(cropped_image)  # Process the first detected card

    print("No ID card detected in the image.")
    return None  # Return None if no ID card is detected or processed


def main():
    # Example usage: Replace "Dina_test.jpg" with your image path
    # Ensure the image "Dina_test.jpg" exists in the same directory or provide the full path.
    image_file = "Dina_test.jpg"  # Make sure this image exists for testing

    # Check if reader initialized correctly
    if reader is None:
        print("Cannot run main: EasyOCR Reader failed to initialize.")
        return

    print(f"Processing image: {image_file}")
    result_data = detect_and_process_id_card(image_file)

    if result_data:
        if len(result_data) == 7:  # Corresponds to (nid, birth_date, governorate, gender, conf, full_name, address)
            id_number, birth_date, governorate, gender, conf_score, full_name_val, address_val = result_data
            print("\n--- Extracted ID Card Front Information ---")
            print(f"National ID: {id_number}")
            print(f"Confidence Score (NID detection): {conf_score}%")
            print(f"Full Name: {full_name_val}")
            print(f"Address: {address_val}")
            print(f"Birth Date: {birth_date}")
            print(f"Governorate: {governorate}")
            print(f"Gender: {gender}")
        elif len(result_data) == 5:  # Corresponds to (job, expiry, status, issue, expiry_conf)
            job_val, expiry_val, status_val, issue_val, expiry_conf_score = result_data
            print("\n--- Extracted ID Card Other Information ---")
            print(f"Job: {job_val}")
            print(f"Expiry Date: {expiry_val}")
            print(f"Marital Status/Demo: {status_val}")
            print(f"Issue Place/Date: {issue_val}")
            # print(f"Expiry Confidence: {expiry_conf_score}") # expiry_conf is currently always 0.0
        else:
            print("\n--- Unexpected result format ---")
            print(result_data)
    else:
        print("\nNo information extracted or ID card not detected.")


if __name__ == "__main__":
    main()