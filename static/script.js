async function predict() {

    let symbol = document.getElementById("symbol").value.trim().toUpperCase();

    if(symbol === ""){
        alert("Please enter a stock symbol.");
        return;
    }

    document.getElementById("result").innerHTML =
    "<h3>Predicting...</h3>";

    const response = await fetch("/predict",{

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify({
            symbol:symbol
        })

    });

    const result = await response.json();

    if(result.error){
        document.getElementById("result").innerHTML =
        "<span style='color:red;'>"+result.error+"</span>";
        return;
    }

    document.getElementById("result").innerHTML = `
        <h3>Prediction Result</h3>

        <b>Stock:</b> ${result["Stock"]}<br>

        <b>Latest Closing Price:</b> $${result["Latest Close"]}<br>

        <b>Predicted Next Closing Price:</b> $${result["Predicted Next Close"]}<br>
    `;
}