# app/__init__.py

from flask import Flask
from config import Config
from flask_moment import Moment
from flask_bootstrap import Bootstrap
from flask import render_template

bootstrap = Bootstrap()
moment = Moment()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.static_folder = config_class.STATIC_FOLDER
    app.template_folder = config_class.TEMPLATE_FOLDER

    bootstrap.init_app(app)
    moment.init_app(app)

    # Import and register the Blueprint
    from app.Controller.errors import bp_errors as errors
    app.register_blueprint(errors)
    from app.Controller.routes import bp_routes as routes
    app.register_blueprint(routes)

    @app.errorhandler(403)
    def forbidden(error):
        return render_template('403.html'), 403

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        return render_template('500.html'), 500

    @app.errorhandler(401)
    def unauthorized(error):
        return render_template('401.html'), 401

    @app.errorhandler(410)
    def gone(error):
        return render_template('410.html'), 410

    if not app.debug and not app.testing:
        pass
        # ... no changes to logging setup

    return app
