var ttid = "charlottesweb";
var response = $.getJSON("https://query.yahooapis.com/v1/public/yql?q=select%20content%20from%20html%20where%20url%3D%22https%3A%2F%2Ftwitter.com%2F"+ttid+"%22%20and%20xpath%3D'%2F%2Fli%5Bcontains(%40class%2C%22ProfileNav-item--followers%22)%5D%2Fa%2Fspan%5Bcontains(%40class%2C%22ProfileNav-value%22)%5D'&format=json&callback=");
return response.success(function (followers) {
    return followers;
});
console.log(followers)