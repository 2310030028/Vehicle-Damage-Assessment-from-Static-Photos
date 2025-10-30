import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import os
from car_damage_detector import CarDamageDetector
from utils import generate_damage_report
import pandas as pd
import random

# Set page config
st.set_page_config(
    page_title="🚗 Car Damage Assessment AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .damage-item {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .damage-item:hover {
        background-color: #f0f2f6;
        transform: translateX(5px);
    }
    .damage-item.selected {
        background-color: #e6f7ff;
        border-left: 4px solid #1890ff;
    }
    .severity-high { color: #ff4d4f; }
    .severity-medium { color: #faad14; }
    .severity-low { color: #52c41a; }
    .stApp {
        max-width: 1600px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

def get_severity_color(severity):
    return {
        'severe': 'severity-high',
        'moderate': 'severity-medium',
        'minor': 'severity-low'
    }.get(severity.lower(), '')

def estimate_repair_cost(damage):
    """Estimate repair cost based on damage type and severity"""
    base_costs = {
        'scratch': 150,
        'scrape': 300,
        'dent': 500
    }
    
    severity_multiplier = {
        'minor': 0.7,
        'moderate': 1.0,
        'severe': 1.8
    }
    
    base = base_costs.get(damage['type'].lower(), 200)
    multiplier = severity_multiplier.get(damage['severity'].lower(), 1.0)
    
    # Add some randomness to make it more realistic (±20%)
    variation = random.uniform(0.8, 1.2)
    return int(base * multiplier * variation)

def main():
    st.title("🚗 Car Damage Assessment AI")
    
    # Initialize session state
    if 'damage_info' not in st.session_state:
        st.session_state.damage_info = None
    if 'selected_damage' not in st.session_state:
        st.session_state.selected_damage = None
    if 'result_image' not in st.session_state:
        st.session_state.result_image = None
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "jpeg", "png"])
        
        if st.session_state.damage_info and st.session_state.damage_info['damage_detected']:
            st.download_button(
                label="📄 Download Full Report",
                data=generate_damage_report(st.session_state.damage_info),
                file_name="damage_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Initialize detector
    detector = CarDamageDetector()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if uploaded_file is not None:
            # Read and process the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect damage if not already done or if image changed
            if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
                with st.spinner('🔍 Analyzing image for damage...'):
                    result_image, damage_info = detector.detect_damage(image)
                    st.session_state.damage_info = damage_info
                    st.session_state.result_image = result_image
                    st.session_state.last_uploaded = uploaded_file.name
                    st.session_state.selected_damage = None
            
            # Display the result image with hover effects
            if st.session_state.result_image is not None:
                st.image(
                    st.session_state.result_image, 
                    caption='Damage Detection Result', 
                    use_container_width=True
                )
                
                # Add mouse hover effect using HTML/JS
                if st.session_state.damage_info and st.session_state.damage_info['damage_detected']:
                    st.markdown("""
                        <style>
                        .damage-highlight {
                            transition: all 0.3s ease;
                        }
                        .damage-highlight:hover {
                            opacity: 0.8;
                            transform: scale(1.01);
                        }
                        </style>
                    """, unsafe_allow_html=True)
        else:
            # Show welcome/instruction screen
            st.markdown("""
                <div style='text-align: center; padding: 2rem;'>
                    <h2>Welcome to Car Damage Assessment AI</h2>
                    <p>Upload an image of a car to detect and analyze any damages.</p>
                    <p>🔄 Click on damage items in the sidebar to highlight them on the image.</p>
                    <p>💾 Download a detailed report after analysis.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Sidebar content
    with col2:
        if st.session_state.damage_info:
            if st.session_state.damage_info['damage_detected']:
                st.subheader("🔍 Damage Summary")
                
                # Overall statistics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total Damages", st.session_state.damage_info['damage_count'])
                with col_b:
                    severity = st.session_state.damage_info['severity']
                    st.metric("Overall Severity", f"{severity.capitalize()}")
                
                st.markdown("---")
                st.subheader("🔧 Damage Details")
                
                # Damage list
                for i, damage in enumerate(st.session_state.damage_info['damage_areas'], 1):
                    severity_class = get_severity_color(damage['severity'])
                    is_selected = st.session_state.selected_damage == damage['id']
                    
                    # Create a unique key for each damage item
                    damage_key = f"damage_{i}_{damage['id']}"
                    
                    # Calculate estimated cost
                    cost = estimate_repair_cost(damage)
                    
                    # Create a clickable damage item
                    clickable = st.checkbox(
                        f"{i}. {damage['type'].capitalize()}", 
                        key=f"cb_{damage_key}",
                        value=is_selected,
                        label_visibility="collapsed"
                    )
                    
                    # Display damage details
                    with st.container():
                        severity_icon = {
                            'severe': '🔥',
                            'moderate': '⚠️',
                            'minor': 'ℹ️'
                        }.get(damage['severity'].lower(), 'ℹ️')
                        
                        # Calculate estimated cost for this damage
                        cost = estimate_repair_cost(damage)
                        
                        # Create a container for the damage item
                        damage_container = st.container()
                        with damage_container:
                            cols = st.columns([1, 3, 2])
                            with cols[0]:
                                st.markdown(f"**{severity_icon} {damage['severity'].capitalize()}**")
                            with cols[1]:
                                st.markdown(f"{damage['type'].capitalize()}")
                            with cols[2]:
                                st.markdown(f"${cost:,.0f}")
                        
                        # Add click handler
                        if clickable:
                            if st.session_state.selected_damage == damage['id']:
                                st.session_state.selected_damage = None
                            else:
                                st.session_state.selected_damage = damage['id']
                        
                        st.markdown("---")
                        
                # Add report section
                st.markdown("## 📝 Damage Report")
                if st.button("Generate Full Report"):
                    report = generate_damage_report(st.session_state.damage_info)
                    st.download_button(
                        label="📥 Download Full Report",
                        data=report,
                        file_name="damage_report.txt",
                        mime="text/plain"
                    )
                    st.text_area("Report Preview", report, height=300)
                
                # Add total cost
                if st.session_state.damage_info['damage_detected']:
                    total_cost = sum(estimate_repair_cost(d) for d in st.session_state.damage_info['damage_areas'])
                    st.markdown(f"### 💰 Estimated Total Repair Cost: **${total_cost:,.2f}**")
                    
                    # Add insurance information
                    st.markdown("""
                    ### 📋 Insurance Information
                    - This is an estimate only. Actual repair costs may vary.
                    - Contact your insurance provider for coverage details.
                    - Keep all receipts and documentation for claims.
                    """)
                
                # Add JavaScript for hover effect
                st.markdown("""
                <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const damages = document.querySelectorAll('.damage-item');
                    damages.forEach(damage => {
                        damage.addEventListener('mouseenter', function() {
                            const damageId = this.getAttribute('data-damage-id');
                            // Highlight corresponding damage on image
                            // This would require custom JavaScript to work with Streamlit
                        });
                        damage.addEventListener('mouseleave', function() {
                            // Remove highlight
                        });
                    });
                });
                </script>
                """, unsafe_allow_html=True)
                
            else:
                st.success("✅ No damage detected in the image.")
                st.markdown("### 🎉 Your car looks good!")
                st.markdown("No visible damage was detected in the uploaded image.")
                
        elif uploaded_file is not None:
            st.warning("No damage detected in the uploaded image.")
            
    # Add some space at the bottom
    st.markdown("---")
    st.markdown("""
    ### About
    This tool uses computer vision to detect and assess car damage. 
    For best results, use well-lit photos taken from straight angles.
    """, unsafe_allow_html=True)
    
    # Add damage item display if damage is detected
    if 'damage_info' in st.session_state and st.session_state.damage_info and st.session_state.damage_info['damage_detected']:
        st.markdown("### Detected Damages")
        for damage in st.session_state.damage_info['damage_areas']:
            severity_icon = {
                'severe': '🔥',
                'moderate': '⚠️',
                'minor': 'ℹ️'
            }.get(damage['severity'].lower(), 'ℹ️')
            
            severity_class = {
                'severe': 'severity-high',
                'moderate': 'severity-medium',
                'minor': 'severity-low'
            }.get(damage['severity'].lower(), '')
            
            cost = estimate_repair_cost(damage)
            
            st.markdown(f"""
            <div class='damage-item' style="margin: 10px 0; padding: 10px; border-radius: 5px; border-left: 4px solid #1890ff;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class='{severity_class}'>
                        {severity_icon} {damage['severity'].capitalize()} - {damage['type'].capitalize()}
                    </div>
                    <div style="font-weight: bold; color: #1890ff;">
                        ${cost:,}
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 0.25rem;">
                    Area: {int(damage['area'])} px² • Location: ({damage['x']}, {damage['y']})
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add some space between items
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            
        # Total estimated cost
        if st.session_state.damage_info['damage_areas']:
            total_cost = sum(estimate_repair_cost(d) for d in st.session_state.damage_info['damage_areas'])
            st.markdown("---")
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; font-size: 1.1rem; font-weight: bold; padding: 0.5rem 0;">
                    <span>Total Estimated Cost:</span>
                    <span>${total_cost:,}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ No damage detected in the image.")
            st.balloons()

if __name__ == "__main__":
    main()