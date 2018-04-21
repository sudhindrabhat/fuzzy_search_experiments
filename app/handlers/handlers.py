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
        print self.settings.keys()
        self.fuzzy_search = self.settings.fuzzy_search

    def get(self, *args, **kwargs):
        query = self.get_argument('query', '')
        if not query:
            raise InvalidInput('query cannot be empty')
        view = JsonView({'api_access_key': self.fuzzy_search.test_access, 'query':query}).render()
        self.finish(view)

class GetApiAccessKeyHandler(BaseHandler):
    def get(self, *args, **kwargs):
        view = JsonView({'api_access_key': 'abcd'}).render()
        self.finish(view)


class HttpNotFoundHandler(BaseHandler):
    def prepare(self):
        raise HTTPError(404)


