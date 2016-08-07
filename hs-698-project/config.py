import os


class BaseConfig(object):

    basedir=os.path.abspath(os.path.dirname(__file__))
    db_path=os.path.join(basedir, 'api', 'dataset', 'cms.db')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + db_path #required by Flask-SQLAlchemy extension -- path to db file
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(BaseConfig):
    DEBUG=True
    SQLALCHEMY_DATABASE_URI = "postgresql://postgres:abcd1234@localhost/cms_post"


class ProductionConfig(BaseConfig):
    DEBUG=False
    # SQLALCHEMY_DATABASE_URI = "postgresql://postgres1:abcd1234@localhost/cms_post1"
    # SQLALCHEMY_DATABASE_URI = "postgresql://pg:abcd1234@hs698-cms.ckop501geaet.us-west-2.rds.amazonaws.com:5432/cms_post"
    SQLALCHEMY_DATABASE_URI = "postgresql://pg:abcd1234@hs698-cmsv2.ckop501geaet.us-west-2.rds.amazonaws.com:5432/cms_postv2"