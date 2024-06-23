from flask import Flask

app = Flask(__name__)


@app.route("/")
def welcome():
    return "Welcome to my website hellow hee "


@app.route("/members")
def members():
    return "Welcome Members , this is an members site"


if __name__ == "__main__":
    app.run(debug=True)
