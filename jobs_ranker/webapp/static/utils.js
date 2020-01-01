function clickButtonOnEnter(buttonId) {
    return function (event) {
        if (event.keyCode == 13) {
            event.preventDefault();
            document.getElementById(buttonId).click();
        }
    };
}
