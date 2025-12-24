import cv2 
import sys
import os
import numpy as np
from datetime import datetime
import pickle

def initialize_face_recognizer():
    """
    Initialize face recognizer and create face database
    """
    # different face recognizers 
    # LBPH is usually most robust for varied lighting and conditions
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Alternative recognizers (uncomment to use)
    # recognizer = cv2.face.EigenFaceRecognizer_create()
    # recognizer = cv2.face.FisherFaceRecognizer_create()
    
    return recognizer


def train_face_recognizer(recognizer, data_dir='face_data'):
    """
    Train the face recognizer with known faces
    """
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        print("Add face images in subfolders (folder name = person's name)")
        return recognizer, {}, False
    
    # Check if I have the training data
    if len(os.listdir(data_dir)) == 0:
        print(f"No training data found in {data_dir}")
        return recognizer, {}, False
    
    # Load face cascade for preprocessing
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Traverse through face data directory
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if dir_name not in label_ids:
                label_ids[dir_name] = current_id
                current_id += 1
            
            label_id = label_ids[dir_name]
            person_dir = os.path.join(root, dir_name)
            
            for filename in os.listdir(person_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, filename)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Detect and preprocess face
                        detected_faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(30, 30)
                        )
                        
                        for (x, y, w, h) in detected_faces:
                            face_roi = gray[y:y+h, x:x+w]
                            # Resize face to consistent size
                            face_resized = cv2.resize(face_roi, (200, 200))
                            faces.append(face_resized)
                            labels.append(label_id)
    
    if len(faces) == 0:
        print("No faces found in training data")
        return recognizer, {}, False
    
    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    
    # Save the trained model
    recognizer.save('face_recognizer_model.yml')
    
    # Save label mappings
    with open('label_mappings.pkl', 'wb') as f:
        pickle.dump(label_ids, f)
    
    print(f"Trained on {len(faces)} face samples for {len(label_ids)} people")
    return recognizer, label_ids, True

def capture_new_face(name, data_dir='face_data'):
    """
    Capture new face samples for training
    """
    # Create person's directory
    person_dir = os.path.join(data_dir, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    print(f"Capturing face samples for {name}")
    print("Look at the camera and move your head slightly")
    print("Press 'c' to capture, 'q' to quit")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    sample_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangle around face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Samples: {sample_count}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f'Training: {name}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Capture Face Samples', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and len(faces) == 1:
            # Save face sample
            face_roi = gray[y:y+h, x:x+w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}_{sample_count}.jpg"
            filepath = os.path.join(person_dir, filename)
            
            # Save the full frame with face marked
            cv2.imwrite(filepath, frame)
            sample_count += 1
            print(f"Saved sample {sample_count}: {filename}")
            
            if sample_count >= 20:  # Collect 20 samples
                print("Collected enough samples")
                break
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {sample_count} samples for {name}")
    return sample_count > 0

def track_face_with_recognition():
    """
    Open camera and track/recognize faces in real-time
    """
    # Initialize face recognizer
    recognizer = initialize_face_recognizer()
    label_ids = {}
    recognition_enabled = False
    
    # Try to load existing model
    if os.path.exists('face_recognizer_model.yml') and os.path.exists('label_mappings.pkl'):
        try:
            recognizer.read('face_recognizer_model.yml')
            with open('label_mappings.pkl', 'rb') as f:
                label_ids = pickle.load(f)
            recognition_enabled = True
            print(f"Loaded trained model with {len(label_ids)} known faces")
        except:
            print("Could not load existing model, starting fresh")
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        sys.exit(1)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*50)
    print("FACE TRACKING & RECOGNITION SYSTEM")
    print("="*50)
    print("Commands:")
    print("  'q' - Quit")
    print("  's' - Take screenshot")
    print("  't' - Train/re-train recognizer")
    print("  'a' - Add new face to database")
    print("  'r' - Toggle recognition on/off")
    print("="*50)
    
    if recognition_enabled:
        print(f"Recognition: ENABLED ({len(label_ids)} known faces)")
    else:
        print("Recognition: DISABLED (No trained model)")
    
    frame_count = 0
    confidence_threshold = 70  # Lower is more confident
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            color = (0, 255, 0)  # Green for unknown
            label = "Unknown"
            confidence = 0
            
            if recognition_enabled:
                try:
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (200, 200))
                    
                    # Predict using recognizer
                    label_id, confidence = recognizer.predict(face_resized)
                    
                    # Find name for the label_id
                    for name, id_num in label_ids.items():
                        if id_num == label_id and confidence < confidence_threshold:
                            label = name
                            color = (255, 0, 0)  # Blue for recognized
                            break
                except:
                    pass
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Display name and confidence
            display_text = f"{label}"
            if recognition_enabled and label != "Unknown":
                display_text += f" ({100 - confidence:.1f}%)"
            
            cv2.putText(frame, display_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw face center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Display coordinates for first face only
            if faces[0][0] == x and faces[0][1] == y:
                info_text = f"X: {center_x}, Y: {center_y}, Size: {w}x{h}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display face count
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display recognition status
        status = "ON" if recognition_enabled else "OFF"
        status_color = (0, 255, 0) if recognition_enabled else (0, 0, 255)
        cv2.putText(frame, f'Recognition: {status}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display known faces count
        if recognition_enabled:
            cv2.putText(frame, f'Known: {len(label_ids)}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display frame count
        frame_count += 1
        cv2.putText(frame, f'Frame: {frame_count}', (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display command help at bottom
        cv2.putText(frame, "Commands: q=Quit, s=Save, t=Train, a=Add, r=Toggle", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Face Tracker with Recognition', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_name = f"face_recognition_{timestamp}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved as {screenshot_name}")
        elif key == ord('t'):
            print("\nTraining face recognizer...")
            recognizer, new_label_ids, success = train_face_recognizer(recognizer)
            if success:
                label_ids = new_label_ids
                recognition_enabled = True
                print(f"Training complete! Model now knows {len(label_ids)} people.")
            else:
                print("Training failed. Make sure you have face images in 'face_data' folder.")
        elif key == ord('a'):
            name = input("\nEnter name for new face: ").strip()
            if name:
                if capture_new_face(name):
                    print("Now re-training with new face...")
                    recognizer, label_ids, success = train_face_recognizer(recognizer)
                    if success:
                        recognition_enabled = True
                        print(f"Added {name} to known faces. Total: {len(label_ids)} people.")
        elif key == ord('r'):
            recognition_enabled = not recognition_enabled
            status = "ENABLED" if recognition_enabled else "DISABLED"
            print(f"\nRecognition {status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nFace tracking stopped")

def setup_face_database():
    """
    Interactive setup for face database
    """
    print("\nFACE DATABASE SETUP")
    print("="*50)
    
    data_dir = 'face_data'
    
    if os.path.exists(data_dir):
        print(f"Existing face database found in '{data_dir}'")
        print("Contents:")
        for item in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, item)):
                face_count = len([f for f in os.listdir(os.path.join(data_dir, item)) 
                                if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  - {item}: {face_count} samples")
    else:
        print(f"No existing database found. Will create '{data_dir}' folder.")
    
    print("\nOptions:")
    print("1. Add new person to database")
    print("2. Continue to face tracking")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        name = input("Enter person's name: ").strip()
        if name:
            capture_new_face(name)
            setup_face_database()
    elif choice == '2':
        return
    else:
        print("Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    # Check and install required packages
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "opencv-python", "numpy"])
        print("Please run the script again.")
        sys.exit(1)
    
    # Setup face database
    setup_face_database()
    
    # Start face tracking with recognition
    track_face_with_recognition()