# app/Controller/routes.py
import sys
from flask import Blueprint, render_template, flash, redirect, url_for, request
from config import Config

bp_routes = Blueprint('routes', __name__, template_folder=Config.TEMPLATE_FOLDER)


@bp_routes.route('/', methods=['GET'])
def index():
    return render_template('index.html', title="Tuber Ruler")
