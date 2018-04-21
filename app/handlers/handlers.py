import hashlib
import json
from random import randint
from time import sleep
import re
from tornado.web import HTTPError
from app.exception.customexceptions import InvalidInput, InternalError
from app.handlers.base import BaseHandler
from app.model.search_data import SearchModel
from app.view.templates.json.base import JsonView
from lib.fuzzy_search_expt import MyFuzzySearch
from debug_config import Config

class SearchHandler(BaseHandler):
    def initialize(self):
        #todo: maybe init it in model?
        #self.fuzzy_search = MyFuzzySearch(max_lev_distance=2)
        #self.fuzzy_search.create_dictionary(Config.fname)
        print 'init!!!!!!!!!!'
        self.fuzzy_search = 'hello world'

    def get(self, *args, **kwargs):
        query = self.get_argument('query', '')
        view = JsonView({'api_access_key': self.fuzzy_search, 'query':query}).render()
        self.finish(view)

class GetApiAccessKeyHandler(BaseHandler):
    def get(self, *args, **kwargs):
        view = JsonView({'api_access_key': 'abcd'}).render()
        self.finish(view)


class HttpNotFoundHandler(BaseHandler):
    def prepare(self):
        raise HTTPError(404)


