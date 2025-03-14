#%%
import sys
import os
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from flask import Flask, render_template, jsonify, request, send_file
import plotly.express as px
from datetime import datetime
import importlib
import json
import webbrowser
from threading import Timer

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)


# function updateGridView() {
#     fetch('/get_grid_view')
#         .then(response => response.json())
#         .then(data => {
#             Plotly.newPlot('grid-view', [{
#                 z: data.grid_data,
#                 type: 'heatmap',
#                 colorscale: 'Jet'
#             }], {
#                 title: 'Grid View of All Frames',
#                 shapes: [{
#                     type: 'rect',
#                     x0: data.highlight_x0,
#                     y0: data.highlight_y0,
#                     x1: data.highlight_x1,
#                     y1: data.highlight_y1,
#                     line: {
#                         color: 'red',
#                         width: 2
#                     }
#                 }]
#             });
#         });
# }


app = Flask(__name__)

# Global variables to store data
class DataStore:
    scan_data = None
    base_path = None
    scan_number = None
    current_frame = 666
    results_dir = '/net/micdata/data2/12IDC/2025_Feb/ZCB_9_3D_/'
    base_path = '/net/micdata/data2/12IDC/2025_Feb/ptycho/'
    scan_number = 5102
    #reconstruction_name = 'Ndp128_LSQML_c1000_m0.5_p15_cp_mm_opr3_ic_pc_ul2'
    reconstruction_name = 'Ndp128_LSQML_c1000_m0.5_p10_cp_mm_opr3_ic_pc_ul2'
    center_x = 517  # Default values from pty_chi_result_loader
    center_y = 575
    dpsize = 128
    grid_size_row = 29
    grid_size_col = 36

data_store = DataStore()

# Create templates directory and HTML file
template_dir = Path(__file__).parent / 'templates'
template_dir.mkdir(exist_ok=True)

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Scan Viewer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .grid-container {
            display: grid;
            grid-template-columns: 1fr 2fr 2fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .plot {
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
        .controls {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
        }
        .info-panel {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="grid-container">
        <div class="controls">
            <h3>File and Experiment Information</h3>
            <div>
                <label for="base-path">Base Path:</label>
                <input type="text" id="base-path" style="width: 80%;">
            </div>
            <div>
                <label for="scan-number">Scan Number:</label>
                <input type="number" id="scan-number" min="0" max="9999" value="0">
            </div>
            <div>
                <label for="center-x">Center X:</label>
                <input type="number" id="center-x" value="517">
                <label for="center-y">Center Y:</label>
                <input type="number" id="center-y" value="575">
            </div>
            <div>
                <label for="dpsize">DP Size:</label>
                <input type="number" id="dpsize" value="128">
            </div>
            <button onclick="loadScan()">Load Scan</button>
            <div class="info-panel" id="info-display"></div>
        </div>

        <!-- Grid View -->
        <div class="plot" id="main-plot"></div>
    </div>

    <script>
        function loadScan() {
            const basePath = document.getElementById('base-path').value;
            const scanNumber = document.getElementById('scan-number').value;
            const centerX = document.getElementById('center-x').value;
            const centerY = document.getElementById('center-y').value;
            const dpsize = document.getElementById('dpsize').value;
            
            fetch('/load_scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    base_path: basePath,
                    scan_number: scanNumber,
                    center_x: centerX,
                    center_y: centerY,
                    dpsize: dpsize
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update info display
                    document.getElementById('info-display').textContent = data.info;
                    
                    // Create plot
                    Plotly.newPlot('main-plot', [{
                        z: data.plot_data,
                        type: 'heatmap',
                        colorscale: 'Jet'
                    }], {
                        title: 'Test Plot',
                        width: 600,
                        height: 600
                    });
                } else {
                    console.error('Error:', data.error);
                }
            });
        }
    </script>
</body>
</html>
"""

with open(template_dir / 'index.html', 'w') as f:
    f.write(html_content)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load_scan', methods=['POST'])
def load_scan():
    try:
        data = request.json
        data_store.base_path = data['base_path']
        data_store.scan_number = int(data['scan_number'])
        data_store.center_x = int(data['center_x'])
        data_store.center_y = int(data['center_y'])
        data_store.dpsize = int(data['dpsize'])
        
        data_store.scan_data = np.sum(ptNN_U.load_h5_scan_to_npy(data_store.base_path,data_store.scan_number,plot=False,point_data=True),axis=0) 
        
        info_text = f"""
        Data Shape: {data_store.scan_data.shape}
        """
        
        # Update the plot immediately after loading
        return jsonify({
            'success': True, 
            'info': info_text,
            'plot_data': data_store.scan_data.tolist()  # Send the data to plot
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# @app.route('/get_grid_view')
# def get_grid_view():
#     if data_store.scan_data is not None:
#         image_size = data_store.scan_data[0].shape
#         grid_image = np.zeros((data_store.grid_size_row * image_size[0], 
#                              data_store.grid_size_col * image_size[1]))
        
        
#         for j in range(data_store.grid_size_row):
#             for i in range(data_store.grid_size_col):
#                 image_idx = j * data_store.grid_size_col + i
#                 if image_idx < len(data_store.scan_data):
#                     grid_image[
#                         j * image_size[0]:(j + 1) * image_size[0],
#                         i * image_size[1]:(i + 1) * image_size[1]
#                     ] = data_store.scan_data[image_idx]
        
#         # Calculate highlight position for current frame
#         row_idx = data_store.current_frame // data_store.grid_size_col
#         col_idx = data_store.current_frame % data_store.grid_size_col
        
#         return jsonify({
#             'grid_data': grid_image.tolist(),
#             'highlight_x0': col_idx * image_size[1],
#             'highlight_y0': row_idx * image_size[0],
#             'highlight_x1': (col_idx + 1) * image_size[1],
#             'highlight_y1': (row_idx + 1) * image_size[0]
#         })
#     return jsonify({'error': 'No data loaded'})


def open_browser():
    webbrowser.open('http://127.0.0.1:5000/')

def main():
    Timer(1, open_browser).start()
    app.run(debug=False)

if __name__ == '__main__':
    main()
# %%
