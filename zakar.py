@pytest.fixture(scope='module')
def text():
    data = ['hallo selamat pagi', 'silahkan transfer ke nomor rekening', 'selamat anda berhasil memnangkan']
    pred = [0, 1, 2]

    return {
        'data': data,
        'pred': pred
    }
