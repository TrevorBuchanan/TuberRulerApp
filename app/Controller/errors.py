from flask import Blueprint, render_template

bp_errors = Blueprint('errors', __name__)


@bp_errors.errorhandler(404)
def not_found_error(error):
    return render_template('404error.html'), 404


@bp_errors.errorhandler(500)
def internal_error(error):
    return render_template('500error.html'), 500
