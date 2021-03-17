const DRINKS = [
"water", // 0
"whisky", // 1
"vodka soda", // 2  
"red wine", // 3
"white wine", // 4
"light beer", // 5
"beer", // 6
"hard seltzer", // 7 
"cosmopolitain", // 8
"manhattan", // 9
"long island", // 10
"martini", // 11
"margarita" // 12
];

const SAMPLE_MOVIES = [
'Avatar, 2009, PG-13, 18 Dec 2009, 162 , Sam Worthington, Zoe Saldana, Sigourney Weaver, Sten Lang Won 3 Oscars. Another 86 wins & 130 ination$760,5625 US0 Feb 20 James Cameron Action, Adventure, Fantasy Sci-Fi English, Spanish 83 A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.',
'Titanic, 1997, PG-13, 19 Dec 1997, 194 , Leonardo DiCaprio, Kate Winslet, Billy Zane, Kathy Ba Won 11 Oscars. Another 112 wins & 83 nominatio $659,363, USA, Mex 01 Jun 2 James Cameron Drama, Roma English, Swedish, Italian, Fre A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.',
'Black Panther, 2018, PG-13, 16 Feb 2018, 134 , Chadwick Boseman, Michael B. Jordan, Lupita Nyongo, Danai Gurir Won 3 Oscars. Another 107 wins & 265 nominations $700,426,56 US 02 May 201 Ryan Coogle Action, Adventure, Sci-F English, Swahili, Nama, Xhosa, Korea 8 TChalla, heir to the hidden but advanced kingdom of Wakanda, must step forward to lead his people into a new future and must confront a challenger from his past.',
'The Dark Knight, 2008, PG-13, 18 Jul 2008, 152 , Christian Bale, Heath Ledger, Aaron Eckhart, Michael Cain Won 2 Oscars. Another 156 wins & 163 nominations $534,858,44 USA, U 14 Jun 201 Christopher Nola Action, Crime, Drama, Thrille English, Mandari 8 When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
'Citizen Kane, 1941, PG, 05 Sep 1941, 119 , Joseph Cotten, Dorothy Comingore, Agnes Moorehead, Ruth Warric Won 1 Oscar. Another 9 wins & 13 nominations $1,585,63 US 28 Jun 201 Orson Welle Drama, Myster English, Italia 10 Following the death of publishing tycoon Charles Foster Kane, reporters scramble to uncover the meaning of his final utterance; Rosebud.',
'Star Wars Episode VII - The Force Awakens, Year 2015, Rated PG-13, Released 18 Dec 2015, Runtime 138 min, Harrison Ford, Mark Hamill, Carrie Fisher, Adam Drive Nominated for 5 Oscars. Another 61 wins & 126 nominations $936,662,22 US 01 Apr 201 J.J. Abram Action, Adventure, Sci-F Englis 8 As a new threat to the galaxy rises, Rey, a desert scavenger, and Finn, an ex-stormtrooper, must join Han Solo and Chewbacca to search for the one hope of restoring peace.',
];

// SAMPLE_MOVIES[a]:DRINKS[b],
const MOVIE_PAIRS = [
  {'movie':SAMPLE_MOVIES[0],'drink':DRINKS[6]},
  {'movie':SAMPLE_MOVIES[1],'drink':DRINKS[5]},
  {'movie':SAMPLE_MOVIES[2],'drink':DRINKS[3]},
  {'movie':SAMPLE_MOVIES[3],'drink':DRINKS[2]},
  {'movie':SAMPLE_MOVIES[4],'drink':DRINKS[10]},
  {'movie':SAMPLE_MOVIES[5],'drink':DRINKS[3]}
];


const TEST_CASE = 'Avengers Endgame, Year 2019, Rated PG-13, Released 26 Apr 2019, Runtime 181 min, Robert Downey Jr., Chris Evans, Mark Ruffalo, Chris Hemswort Nominated for 1 Oscar. Another 69 wins & 102 nominations $858,373,00 US 30 Jul 201 Anthony Russo, Joe Russ Action, Adventure, Drama, Sci-F English, Japanese, Xhosa, Germa 7 After the devastating events of Avengers Infinity War (2018), the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos actions and restore balance to the universe.';

var textCompareModel = null;

use.load().then(model => {
  textCompareModel = model;
  let btn = document.querySelector('#generateRecomendationBtn');
  btn.disabled = false;
  btn.addEventListener('click', getRecommendation);
  //getRecommendation(TEST_CASE);
});

const getRecommendation = () => {
  const text = document.querySelector("#text-area-1").value;
  const recommedationSpan = document.querySelector("#recommendationSpan");
  recommedationSpan.textContent = "Working ...";
  recommedationSpan.style.color = 'blue';
  if(text){
    drinkRecommendation(text).then(recommedation =>{
      console.log(recommedation);
      recommedationSpan.style.color = 'green';
      recommedationSpan.textContent = recommedation.drink;
    });
  }
}

// Use the TensorFlow Model to compare two text strings,
// return a float that represent how closely (0 to 1) the
// strings match eachother. A 1 is a perfect match. A 0 
// is no match. The function runs ayscronously.
const match = async(text1, text2) => {
  const texts = [text1, text2];
  const embeddings = await textCompareModel.embed(texts);
  //console.log("comparing movies...");
  const text1tf = tf.slice(embeddings, [0, 0], [1]);
  const text2tf = tf.slice(embeddings, [1, 0], [1]);
  return tf.matMul(text1tf, text2tf, false, true).dataSync();
}

// Using the modified Jaccard Index method, 
// generate a drink recommendation base on how 
// well a movie matches other movies from the 
// sample set which have alreay been pair with
// dirnks.
// ModifiedJaccardIndex(inputMovie, inputDrink) = ( sum( MOVIES_PAIRED_SET.forEach(_MOVIE => match(_MOVIE, inputMovie) )) ) / MOVIES_PAIRED_SET.length

const generateRecommendationIndex = async (inputMovie, inputDrink) => {
  //console.log("generateing recommedation for \n" + inputMovie + "\n and \n" + inputDrink);
  let sum = 0;
  let iteration = 0;
  const moviesPairedSet = getMoviePairSet(inputDrink);
  if(moviesPairedSet.length === 0) return 0;
  let matchIndicies = [];
  moviesPairedSet.forEach((moviePair) =>  {
    const matchIndex = Promise.resolve(match(moviePair, inputMovie));
    matchIndicies.push(matchIndex);
  });
  return Promise.all(matchIndicies).then((result) => {
    //console.log(result);
    result.forEach(index => {
      sum = sum + index[0];
    })
    //console.log("final sum: " + sum); 
    const recommedation = {};
    recommedation.drink = inputDrink;
    recommedation.score = sum / moviesPairedSet.length;
    //console.log("final recommedation: " + recommedation);
    return recommedation;
  });
}

const getMoviePairSet = (inputDrink) => {
  const moviePairSet = [];
  MOVIE_PAIRS.forEach(pair => {
    if(pair.drink === inputDrink){
      moviePairSet.push(pair.movie);
    }
  });
  return moviePairSet;
}

const drinkRecommendation = (movieText) => {
  let drinkRecommendations = [];
  DRINKS.forEach(drink => {
    const drinkRecommendation = generateRecommendationIndex(movieText, drink);
    drinkRecommendations.push(drinkRecommendation);
  });
  return Promise.all(drinkRecommendations).then((recommedations) =>{
    //console.log(recommedations);
    let drink = null;
    let score = 0;
    recommedations.forEach(rec => {
      if(rec){
        if(score < rec.score){
          score = rec.score;
          drink = rec.drink;
        }
      }
    });
    const recommedation = {};
    recommedation.score = score;
    recommedation.drink = drink;
    //console.log("For: \n" + TEST_CASE + "\n");
    //console.log("Recommend " + drink + " with " + Math.round(score * 100) + "% match score");
    return recommedation;
  });

}




