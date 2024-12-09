import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    ROOT_PATH = basedir
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    STATIC_FOLDER = os.path.join(basedir, 'app', 'View', 'static')
    TEMPLATE_FOLDER = os.path.join(basedir, 'app', 'View', 'templates')
    UPLOAD_FOLDER = os.path.join(basedir, 'app', 'View', 'uploads')
