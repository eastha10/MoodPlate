from django.shortcuts import render

# Create your views here.
def main(request):
    prompt = request.GET.get('prompt', '')
    print("prompt: ",prompt)
    context = {'prompt': prompt}
    return render(request, 'main.html',context)