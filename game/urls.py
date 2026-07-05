from django.urls import path
from . import views

urlpatterns = [
    # http://localhost:8000/api/play/ でアクセスできるようにする
    path('api/play/', views.get_ai_move, name='get_ai_move'),
    # トップページ (http://localhost:8000/) にアクセスしたら index 関数を呼ぶ
    path('', views.index, name='index'),
    # AIのAPI (すでに書いたもの)
    path('api/play/', views.get_ai_move, name='get_ai_move'),
]