import os
from app.handlers.base import BaseHandler
from app.handlers.handlers import SearchHandler, HttpNotFoundHandler

import tornado.ioloop
import tornado.options
import tornado.web
import tornado.httpserver
from tornado.options import define, options
from tornado.web import url

from lib.fuzzy_search_expt import MyFuzzySearch
from debug_config import Config

define("port", default=8888, type=int)


class Application(tornado.web.Application):
    def __init__(self, *overrides):
        handlers = [
            url(r'/search', SearchHandler),
            url(r'/(.*)', HttpNotFoundHandler)
        ]

        fuzzy_search = MyFuzzySearch(fname=Config.fname, max_lev_distance=2)

        settings = {
            'template_path': os.path.join(os.path.dirname(__file__), 'templates'),
            'xsrf_cookies': False,
            'debug': True,
            'log_file_prefix': "tornado.log",
            'fuzzy_search': fuzzy_search
        }

        tornado.web.Application.__init__(self, handlers, **settings)


def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()
