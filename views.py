
import subprocess
import os
import sys
import random
import string
from skimage import io
from werkzeug.utils import secure_filename
from flask import render_template
from flask import send_from_directory
from flask import Flask, request, redirect, url_for
from app import app
from main import pixel

UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'JPG', 'JPEG'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = os.urandom(64)

def id_gen(size=12, chars=string.ascii_uppercase + string.digits):
    """Return random string for temp files."""
    return "".join(random.choice(chars) for _ in range(size))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def apply_pixel(in_pic, out_pic, params):
    """Apply pixel func to pic."""
    print(params)
    im = io.imread(in_pic)
    pix = pixel(im, magn=params["m"],
                lp=params["l_p"],
                rp=params["r_p"])
    io.imsave(out_pic, pix)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            # get file and save it
            filename = secure_filename(file.filename)
            in_file = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(in_file)

            params = {}
            params["l_p"] = int(request.form["left_percentile"])
            params["r_p"] = int(request.form["right_percentile"])
            params["m"] = True if request.form.getlist("magnify") else False
            # params["i"] = True if request.form.getlist("interlacing") else False
            print(params)
            output_filename = "{}.jpg".format(id_gen(12))
            out_file = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
    
            apply_pixel(in_file, out_file, params)
            # save in the same folder
            # io.imsave(in_file, glim)
            os.remove(in_file)
            return redirect(url_for('uploaded_file',
                                    filename=output_filename))
    return render_template('start_page.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
