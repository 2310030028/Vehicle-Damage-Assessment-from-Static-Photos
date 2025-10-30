import os
import json
from datetime import datetime
from typing import Dict, Any

def generate_damage_report(damage_info: Dict[str, Any]) -> str:
    """
    Generate a detailed text report of the damage assessment.
    
    Args:
        damage_info (dict): Dictionary containing damage assessment information
        
    Returns:
        str: Formatted damage report
    """
    if not damage_info['damage_detected']:
        return "No damage detected in the provided image."
    
    report = []
    report.append("=== VEHICLE DAMAGE ASSESSMENT REPORT ===\n")
    
    # Summary
    report.append("SUMMARY:")
    report.append("-" * 40)
    report.append(f"Damage Detected: {'Yes' if damage_info['damage_detected'] else 'No'}")
    report.append(f"Total Damage Areas: {damage_info['damage_count']}")
    
    # Calculate total damaged area
    total_area = sum(area['area'] for area in damage_info['damage_areas'])
    report.append(f"Total Damaged Area: {total_area:.0f} pixels")
    
    # Get damage type distribution
    damage_types = {}
    for damage in damage_info['damage_areas']:
        d_type = damage['type']
        damage_types[d_type] = damage_types.get(d_type, 0) + 1
    
    report.append("\nDamage Type Distribution:")
    for d_type, count in damage_types.items():
        report.append(f"- {d_type.capitalize()}: {count} area{'s' if count > 1 else ''}")
    
    report.append(f"\nOverall Severity: {damage_info['severity'].upper()}")
    
    # Add severity explanation
    severity_explanation = {
        'minor': 'Minor damage: Small scratches or dents that are primarily cosmetic.',
        'moderate': 'Moderate damage: Noticeable damage that may require repair but is not critical.',
        'severe': 'Severe damage: Significant damage that likely requires immediate attention.'
    }
    report.append(f"\nSeverity Explanation: {severity_explanation.get(damage_info['severity'], 'N/A')}")
    
    report.append("\n" + "="*40 + "\n")
    
    # Detailed damage information
    report.append("DETAILED DAMAGE ANALYSIS:")
    report.append("-" * 40)
    
    for i, damage in enumerate(damage_info['damage_areas'], 1):
        report.append(f"\nDAMAGE AREA {i}:")
        report.append("-" * 30)
        report.append(f"Type: {damage['type'].capitalize()}")
        report.append(f"Severity: {damage.get('severity', 'moderate').capitalize()}")
        report.append(f"Size: {damage['area']:.0f} pixels")
        report.append(f"Location: (X: {damage['x']}, Y: {damage['y']})")
        report.append(f"Dimensions: {damage['width']} x {damage['height']} pixels")
        
        # Add recommendations based on damage type and severity
        recommendations = {
            'scratch': 'May be repairable with paint touch-up or polishing.',
            'dent': 'May require paintless dent repair or traditional bodywork.',
            'scrape': 'May need touch-up paint or panel repair depending on depth.'
        }
        report.append(f"\nRecommended Action: {recommendations.get(damage['type'], 'Inspection recommended.')}")
    
    # Add timestamp and footer
    from datetime import datetime
    report.append("\n" + "="*40)
    report.append("NOTES:")
    report.append("- This is an automated assessment. For accurate evaluation, consult a professional.")
    report.append("- Repair costs may vary based on vehicle make, model, and location.")
    report.append("\n" + "="*40)
    report.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*40)
    
    # Add estimated repair information based on severity
    severity = damage_info.get('severity', 'none').lower()
    report.append("\nESTIMATED REPAIR INFORMATION:")
    report.append("-" * 40)
    
    if severity == 'severe':
        report.append("• Extensive repairs needed")
        report.append("• Potential structural damage")
        report.append("• Estimated repair cost: $$$$")
    elif severity == 'moderate':
        report.append("• Significant repairs needed")
        report.append("• May require professional attention")
        report.append("• Estimated repair cost: $$$")
    elif severity == 'minor':
        report.append("• Minor cosmetic damage")
        report.append("• May be repairable with basic tools")
        report.append("• Estimated repair cost: $")
    
    return "\n".join(report)
    return "\n".join(report)

def save_detection_results(image_path: str, damage_info: Dict[str, Any], 
                          output_dir: str = "outputs") -> str:
    """
    Save detection results to a JSON file.
    
    Args:
        image_path: Path to the original image
        damage_info: Dictionary containing damage assessment
        output_dir: Directory to save results to
        
    Returns:
        str: Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data to save
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{base_name}_{timestamp}.json")
    
    # Add metadata
    result = {
        "image_path": image_path,
        "timestamp": timestamp,
        "detection_results": damage_info
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return output_file

def load_image(image_path: str) -> tuple:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (image_array, image_pil)
    """
    import cv2
    from PIL import Image
    
    # Read with OpenCV
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img_cv)
    
    return img_cv, img_pil
