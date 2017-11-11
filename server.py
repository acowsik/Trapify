from bottle import post, request, static_file, abort, route, redirect
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


@route('/<filename>')
def get_file():
    if not os.path.samefile(ROOT,
        os.path.commonprefix([ROOT,
            os.path.normpath(os.path.join(ROOT, filename))])):
        abort(404)  # user attempted to access file outside directory

    if filename.endswith('.html') or filename.endswith('.css') or filename.endswith('.js'):
        return static_file(filename, root=ROOT)  # only return html/css/js files


@post('/')
def trapify_file():
    file = request.forms.get("musicfile")
    file = convert_mp3_to_wav(file)
    file = trapify(file)
    file = convert_wav_to_mp3(file)
    return file


def convert_mp3_to_wav(file):
    return file


def trapify(file):
    return file


def convert_wav_to_mp3(file):
    return file
