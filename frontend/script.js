let buttons = document.querySelectorAll("button.btn-primary")

Array.from(buttons).forEach(button => {
    button.addEventListener('click', function (e) {
    e.target.disabled = true
    let request = new XMLHttpRequest();   // new HttpRequest instance
    let url = ''
        let values = {}
    switch (document.querySelector(".active").id){
        case "nav-classification":
            url = "api/classification"
            values = {
               "model": document.querySelector(".active input.model-input").value,
               "text": document.querySelector(".active input.text-input").value
            }
            break;
        case "nav-ner":
            url = "api/ner"
            values = {
               "model": document.querySelector(".active input.model-input").value,
               "text": document.querySelector(".active input.text-input").value
            }
            break;
        case "nav-similarity":
            url = "api/sentence_sim"
            let sentences = [];
            let inps = document.querySelectorAll(".active input.text-input")
            Array.from(inps).forEach(inp => {sentences.push(inp.value);});

            values = {
               "model": document.querySelector(".active input.model-input").value,
               "sentences": sentences
            }
            break;
        case "nav-qa":
            url = "api/qa"
            values = {
               "model": document.querySelector(".active input.model-input").value,
               "question": document.querySelector(".active input.question-input").value,
                "context": document.querySelector(".active input.context-input").value
            }
            break;
        case "nav-generation":
            url = "api/generation"
            values = {
               "model": document.querySelector(".active input.model-input").value,
               "text": document.querySelector(".active input.text-input").value
            }
            break;
        case "nav-seq2seq":
            url = "api/seq2seq"
            values = {
               "model": document.querySelector(".active input.model-input").value,
               "text": document.querySelector(".active input.text-input").value
            }
            break;
        case "nav-translation":
            url = "api/translation"
            let from = document.querySelector("#from-lang").value
            let to = document.querySelector("#to-lang").value
            values = {
               "model": document.querySelector(".active input.model-input").value,
               "text": document.querySelector(".active input.text-input").value,
                "from": from,
                "to": to
            }
            break;
        case "nav-summarization":
            url = "api/summarization"
            values = {
               "model": document.querySelector(".active input.model-input").value,
               "text": document.querySelector(".active input.text-input").value
            }
            break;
        case "nav-qa-closed-book":
            url = "api/closed-book"
            values = {
               "model": document.querySelector(".active input.model-input").value,
               "text": document.querySelector(".active input.text-input").value
            }
            break;

    }

    request.open("POST", url);
    request.setRequestHeader("Content-Type", "application/json");
    request.onload = function () {
        let status = request.status;
        if (status === 200) {
            document.querySelector(".active .output").textContent = JSON.stringify(JSON.parse(request.response),
                null, 2)
            // let answerDiv = document.getElementById("answer-div");
            // let buttonP = document.getElementById('button-p')
            // answerDiv.innerHTML = '';
            // buttonP.innerHTML = '';
            // JSON.parse(request.response).forEach(function (value, index, array) {
            //
            //     let button = document.getElementById("button-collapse-template");
            //     button = button.content.cloneNode(true);
            //     button.querySelector('a').textContent += (index + 1)
            //     button.querySelector('a').setAttribute('aria-controls', 'collapse-id-' + index);
            //     button.querySelector('a').setAttribute('href', '#collapse-id-' + index)
            //     let divParagraph = document.getElementById("card-template");
            //     divParagraph = divParagraph.content.cloneNode(true);
            //     divParagraph.querySelector('.card-body').textContent += value
            //     divParagraph.querySelector('.collapse').setAttribute('id', 'collapse-id-' + index)
            //     divParagraph.querySelector('.header-solve').textContent += (index + 1)
            //     document.getElementById('button-p').appendChild(button);
            //     document.getElementById('answer-div').appendChild(divParagraph);
            // })
            // document.getElementById("answer-div").innerText = JSON.parse(request.response).join(" <br> ");
        } else {
            console.log(status);
        }
        e.target.disabled = false
    };


    request.send(JSON.stringify(values));
})

});
