## Integrate HTML with Flask.
## HTTP verb GET and POST.


from flask import Flask, redirect, url_for, render_template

app = Flask(__name__)


@app.route("/")
def welcome():
    return render_template("index.html")


@app.route("/success/<int:score>")
def success(score):
    return "The Person has Passed and the Score is : " + str(score)


@app.route("/fail/<int:score>")
def fail(score):
    return "The Person has Fail and the Score is : " + str(score)


# Result Checker


@app.route("/results/<int:marks>")
def results(marks):
    result = ""
    if marks < 50:
        result = "fail"
    else:
        result = "success"
    return redirect(
        url_for(result, score=marks)
    )  # this will redirect the page form one page to another


if __name__ == "__main__":
    app.run(debug=True)
