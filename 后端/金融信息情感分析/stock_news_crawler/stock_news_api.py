# -*- coding: utf-8 -*-
"""
Welcome to Noah's Ark
"""
from flask import Flask, jsonify, abort
from Stock_news_collection import StockNewsCollection
app = Flask(__name__)
snc = StockNewsCollection()
def get_stocknews(cd, ns):
    #print(cd,ns)
    err, maxn, res = snc.crawling(cd, ns)
    if err:
        return {'isSuccess':False,
                }
    return {'isSuccess':True, 
            'result':res,
            'maxaccess':maxn}
@app.route('/stocknews/code=<int:cd>&nums=<int:ns>', methods=['GET'])
def get_task(cd, ns):
    try:
        return jsonify(get_stocknews(cd, ns))
    except:
        pass
    abort(404)

    
if __name__ == '__main__':
    app.run()
    #/stocknews/code=688148&nums=5
