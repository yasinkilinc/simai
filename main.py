import sys
import os
import argparse
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from processor import InputProcessor
from reconstruction import FaceReconstructor
from features import FaceFeatures
from interpreter import PhysiognomyInterpreter
from visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="3D Face Physiognomy Analysis")
    parser.add_argument("--source", type=str, help="Path to image or video file. If not provided, runs on dummy data/webcam.")
    args = parser.parse_args()

    processor = InputProcessor(args.source)
    try:
        processor.load_source()
    except Exception as e:
        print(f"Error loading source: {e}")
        return

    reconstructor = FaceReconstructor()
    interpreter = PhysiognomyInterpreter()
    visualizer = Visualizer()

    print(f"Processing source: {args.source if args.source else 'Webcam/Dummy'}")

    for frame in processor.get_frame():
        if frame is None:
            break

        landmarks = reconstructor.process_frame(frame)
        
        if landmarks:
            print("Face detected!")
            # 1. Get 3D Points
            points_3d = reconstructor.get_3d_points(landmarks, frame.shape)
            
            # 2. Extract Features
            face_features = FaceFeatures(points_3d, frame)
            
            # 3. Interpret Personality
            report = interpreter.interpret(face_features)
            
            # 4. Visualize & Save
            annotated_frame = frame.copy()
            visualizer.draw_landmarks(annotated_frame, points_3d)
            visualizer.draw_analysis(annotated_frame, report)
            visualizer.save_image(annotated_frame, "analysis_result.jpg")
            print("Görsel analiz 'analysis_result.jpg' olarak kaydedildi.")

            # 5. Print Report
            print("\n" + "="*30)
            print(f"FİZYONOMİ RAPORU")
            print("="*30)
            print(f"Yüz Şekli: {report['face_shape']}")
            print("-" * 30)
            
            print("\n[+] OLUMLU ÖZELLİKLER:")
            for trait in report['analysis']['positive']:
                print(f"  • {trait}")
            
            print("\n[-] OLUMSUZ / DİKKAT EDİLMESİ GEREKENLER:")
            for trait in report['analysis']['negative']:
                print(f"  • {trait}")
                
            print("\n[?] DİĞER / NÖTR:")
            for trait in report['analysis']['neutral']:
                print(f"  • {trait}")
            print("="*30 + "\n")
            
            # Save to file
            with open("analysis_report.txt", "w", encoding="utf-8") as f:
                f.write("FİZYONOMİ ANALİZ RAPORU\n")
                f.write("="*30 + "\n")
                f.write(f"Yüz Şekli: {report['face_shape']}\n")
                f.write("-" * 30 + "\n")
                f.write("\n[+] OLUMLU ÖZELLİKLER:\n")
                for trait in report['analysis']['positive']:
                    f.write(f"  • {trait}\n")
                f.write("\n[-] OLUMSUZ / DİKKAT EDİLMESİ GEREKENLER:\n")
                for trait in report['analysis']['negative']:
                    f.write(f"  • {trait}\n")
                f.write("\n[?] DİĞER / NÖTR:\n")
                for trait in report['analysis']['neutral']:
                    f.write(f"  • {trait}\n")
            
            print("Rapor 'analysis_report.txt' dosyasına kaydedildi.")
            
            # For now, just break after first face in video to avoid spam
            if not processor.is_image:
                print("Processed one frame from video. Exiting loop for demo.")
                break
        else:
            print("No face detected in this frame.")
            if processor.is_image:
                break

    processor.release()

if __name__ == "__main__":
    main()
