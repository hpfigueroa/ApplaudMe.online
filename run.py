#!/usr/bin/env python
from flaskapp import app
# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app
app.run(host='127.0.0.1', debug = True)