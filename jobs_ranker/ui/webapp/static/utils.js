function clickButtonOnEnter(buttonId) {
    return function (event) {
        if (event.keyCode == 13) {
            event.preventDefault();
            document.getElementById(buttonId).click();
        }
    };
}

function copyToClipboard(element) {
    var $temp = $("<input>");
    $("body").append($temp);
    $temp.val($(element).text()).select();
    document.execCommand("copy");
    $temp.remove();
}
